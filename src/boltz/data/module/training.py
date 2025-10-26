from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pytorch_lightning as pl
import torch
import json
import copy as cp
from numpy import ndarray
from torch import Tensor, from_numpy
from boltz.data.feature.pad import pad_dim
from torch.utils.data import DataLoader
import boltz.data.const as const
from boltz.data.crop.cropper import Cropper
from boltz.data.feature.featurizer import BoltzFeaturizer
from boltz.data.feature.pad import pad_to_max
from boltz.data.feature.symmetry import get_symmetries
from boltz.data.filter.dynamic.filter import DynamicFilter
from boltz.data.sample.sampler import Sample, Sampler
from boltz.data.tokenize.tokenizer import Tokenizer
from boltz.data.types import MSA, Input, Manifest, Record, Structure, AntibodyInfo

@dataclass
class DatasetConfig:
    """Dataset configuration."""

    target_dir: str
    msa_dir: str
    prob: float
    sampler: Sampler
    cropper: Cropper    
    no_msa: Optional[bool] = False
    filters: Optional[list] = None
    train_split: Optional[str] = None
    val_split: Optional[str] = None
    manifest_path: Optional[str] = None


@dataclass
class DataConfig:
    """Data configuration."""

    datasets: list[DatasetConfig] # list of dataset configs
    filters: list[DynamicFilter] # list of dynamic filters
    featurizer: BoltzFeaturizer
    tokenizer: Tokenizer
    max_atoms: int
    max_tokens: int
    max_seqs: int
    samples_per_epoch: int
    batch_size: int
    num_workers: int
    random_seed: int
    distinguish_epitope: bool
    pin_memory: bool
    symmetries: str
    atoms_per_window_queries: int
    min_dist: float
    max_dist: float
    num_bins: int
    no_msa: Optional[bool] = False
    overfit: Optional[int] = None
    pad_to_max_tokens: bool = False
    pad_to_max_atoms: bool = False
    pad_to_max_seqs: bool = False
    crop_validation: bool = False
    return_train_symmetries: bool = False
    return_val_symmetries: bool = True
    train_binder_pocket_conditioned_prop: float = 0.0
    val_binder_pocket_conditioned_prop: float = 0.0
    binder_pocket_cutoff: float = 6.0
    binder_pocket_sampling_geometric_p: float = 0.0
    val_batch_size: int = 1


@dataclass
class Dataset:
    """Data holder."""

    target_dir: Path
    msa_dir: Path
    manifest: Manifest
    prob: float
    sampler: Sampler
    cropper: Cropper
    no_msa: Optional[bool]
    tokenizer: Tokenizer
    featurizer: BoltzFeaturizer


def load_input(record: Record, target_dir: Path, msa_dir: Path, no_msa: bool = False
               ) -> Input:
    """Load the given input data.

    Parameters
    ----------
    record : Record
        The record to load.
    target_dir : Path
        The path to the data directory.
    msa_dir : Path
        The path to msa directory.

    Returns
    -------
    Input
        The loaded input.

    """
    # Load the structure
    structure = np.load(target_dir / "structures" / f"{record.id}.npz")
    structure = Structure(
        atoms=structure["atoms"],
        bonds=structure["bonds"],
        residues=structure["residues"],
        chains=structure["chains"],
        connections=structure["connections"],
        interfaces=structure["interfaces"],
        mask=structure["mask"],
    )

    msas = {}
    if no_msa:
        return Input(structure, msas)
    for chain in record.chains:
        msa_id = chain.msa_id
        # Load the MSA for this chain, if any
        if msa_id != -1:
            msa = np.load(msa_dir / f"{msa_id}.npz")
            msas[chain.chain_id] = MSA(**msa)

    return Input(structure, msas)


def collate(data: list[dict[str, Tensor]]) -> dict[str, Tensor]:
    """Collate the data.

    Parameters
    ----------
    data : list[dict[str, Tensor]]
        The data to collate.

    Returns
    -------
    dict[str, Tensor]
        The collated data.

    """
    # Get the keys
    keys = data[0].keys()

    # Collate the data
    collated = {}
    for key in keys:
        values = [d[key] for d in data]

        if key not in [
            "all_coords",
            "all_resolved_mask",
            "crop_to_all_atom_map",
            "chain_symmetries",
            "amino_acids_symmetries",
            "ligand_symmetries",
        ]:
            # Check if all have the same shape
            shape = values[0].shape
            if not all(v.shape == shape for v in values):
                values, _ = pad_to_max(values, 0)
            else:
                values = torch.stack(values, dim=0)

        # Stack the values
        collated[key] = values

    return collated


def ab_region_type(token: ndarray, spec_mask: ndarray, chain_id: int) -> ndarray:
    """
        Get antibody region label:
        FR1:  1
        CDR1: 2
        FR2:  3
        CDR2: 4
        FR3:  5
        CDR3: 6
        FR4:  7
    """
    indices = [i for i, x in enumerate(token) if x["asym_id"] == chain_id]
    masks = spec_mask[indices]

    diff = np.diff(masks.astype(int))
    diff = np.concatenate(([1], diff))
    segment_ids = np.cumsum(diff != 0)
    
    label = np.zeros_like(spec_mask, dtype=int)
    label[indices] = segment_ids

    return label 

def ag_region_type(token: ndarray, spec_mask: ndarray, ab_chain_ids: list[int], add_epitope: bool = True) -> ndarray: 
    """
        Get antigen region label:
        non-epitope: 8
        epitope: 9
    """
    indices = [i for i, x in enumerate(token) if x["asym_id"] not in ab_chain_ids]
    masks = spec_mask[indices]

    label = np.zeros_like(spec_mask, dtype=int)
    if add_epitope:
        label[indices] = masks + 8
    else:
        label[indices] = 8

    return label

class TrainingDataset(torch.utils.data.Dataset):
    """Base iterable dataset."""

    def __init__(
        self,
        datasets: list[Dataset],
        samples_per_epoch: int,
        symmetries: dict,
        max_atoms: int,
        max_tokens: int,
        max_seqs: int,
        distinguish_epitope: bool = True,
        pad_to_max_atoms: bool = False,
        pad_to_max_tokens: bool = False,
        pad_to_max_seqs: bool = False,
        atoms_per_window_queries: int = 32,
        min_dist: float = 2.0,
        max_dist: float = 22.0,
        num_bins: int = 64,
        overfit: Optional[int] = None,
        binder_pocket_conditioned_prop: Optional[float] = 0.0,
        binder_pocket_cutoff: Optional[float] = 6.0,
        binder_pocket_sampling_geometric_p: Optional[float] = 0.0,
        return_symmetries: Optional[bool] = False,
    ) -> None:
        """Initialize the training dataset."""
        super().__init__()
        self.datasets = datasets
        self.distinguish_epitope = distinguish_epitope
        self.probs = [d.prob for d in datasets]
        self.samples_per_epoch = samples_per_epoch
        self.symmetries = symmetries
        self.max_tokens = max_tokens
        self.max_seqs = max_seqs
        self.max_atoms = max_atoms
        self.pad_to_max_tokens = pad_to_max_tokens
        self.pad_to_max_atoms = pad_to_max_atoms
        self.pad_to_max_seqs = pad_to_max_seqs
        self.atoms_per_window_queries = atoms_per_window_queries
        self.min_dist = min_dist
        self.max_dist = max_dist
        self.num_bins = num_bins
        self.binder_pocket_conditioned_prop = binder_pocket_conditioned_prop
        self.binder_pocket_cutoff = binder_pocket_cutoff
        self.binder_pocket_sampling_geometric_p = binder_pocket_sampling_geometric_p
        self.return_symmetries = return_symmetries
        self.samples = []
        for dataset in datasets:
            records = dataset.manifest.records
            if overfit is not None:
                records = records[:overfit]
            iterator = dataset.sampler.sample(records, np.random)
            self.samples.append(iterator)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        """Get an item from the dataset.

        Parameters
        ----------
        idx : int
            The data index.

        Returns
        -------
        dict[str, Tensor]
            The sampled data features.

        """
        
        dataset_idx = 0
        dataset = self.datasets[dataset_idx]

        # Get a sample from the dataset
        sample: Sample = next(self.samples[dataset_idx])

        # Get the structure
        try:
            input_data = load_input(sample.record, dataset.target_dir, dataset.msa_dir, dataset.no_msa)
        except Exception as e:
            print(
                f"Failed to load input for {sample.record.id} with error {e}. Skipping."
            )
            return self.__getitem__(idx)

        # Tokenize structure
        try:
            tokenized, spec_token_mask = dataset.tokenizer.tokenize(input_data)
        except Exception as e:
            print(f"Tokenizer failed on {sample.record.id} with error {e}. Skipping.")
            return self.__getitem__(idx)

        if isinstance(sample.record.structure, AntibodyInfo):
            h_region_type = ab_region_type(tokenized.tokens, spec_token_mask, sample.record.structure.H_chain_id)
            l_region_type = ab_region_type(tokenized.tokens, spec_token_mask, sample.record.structure.L_chain_id)
            ag_region_types = ag_region_type(tokenized.tokens, spec_token_mask, 
                                            [sample.record.structure.H_chain_id, sample.record.structure.L_chain_id], 
                                            self.distinguish_epitope)
            region_type = h_region_type + l_region_type + ag_region_types
            assert len(region_type) == len(spec_token_mask)

        # Compute crop
        try:
            if self.max_tokens is not None:
                params = {
                    'data': tokenized,
                    'token_mask': spec_token_mask,
                    'token_region': region_type,
                    'random': np.random,
                    'max_atoms': self.max_atoms,
                    'max_tokens': self.max_tokens,
                    'chain_id': sample.chain_id,
                }
                if isinstance(sample.record.structure, AntibodyInfo):
                    params["h_chain_id"] = sample.record.structure.H_chain_id
                    params["l_chain_id"] = sample.record.structure.L_chain_id
                tokenized, spec_token_mask, region_type = dataset.cropper.crop(
                    **params
                )
        except Exception as e:
            print(f"Cropper failed on {sample.record.id} with error {e}. Skipping.")
            return self.__getitem__(idx)

        # Check if there are tokens
        if len(tokenized.tokens) == 0:
            msg = "No tokens in cropped structure."
            raise ValueError(msg)
        
        if isinstance(sample.record.structure, AntibodyInfo):
            indices = [i for i, x in enumerate(tokenized.tokens) if x["asym_id"] in [sample.record.structure.H_chain_id, sample.record.structure.L_chain_id]]
            cdr_token_mask = np.zeros_like(spec_token_mask, dtype=bool)
            cdr_token_mask[indices] = spec_token_mask[indices]
        else:
            indices = [i for i, x in enumerate(tokenized.tokens) if x["asym_id"] == sample.record.structure.pocket_chain_id]
            cdr_token_mask = np.zeros_like(spec_token_mask, dtype=bool)
            cdr_token_mask[indices] = spec_token_mask[indices]
            
        unmask_res_type = cp.deepcopy(tokenized.tokens["res_type"])
        all_masked_res_type = np.where(cdr_token_mask, const.unk_token_ids["PROTEIN"], unmask_res_type)
        tokenized.tokens["res_type"] = all_masked_res_type
        
        if isinstance(sample.record.structure, AntibodyInfo):
            chain_type = torch.ones_like(from_numpy(tokenized.tokens["asym_id"])).long() * 3
            chain_type[tokenized.tokens["asym_id"] == sample.record.structure.H_chain_id] = 1
            chain_type[tokenized.tokens["asym_id"] == sample.record.structure.L_chain_id] = 2
        else:
            chain_type = torch.ones_like(from_numpy(tokenized.tokens["asym_id"])).long()
            chain_type[tokenized.tokens["asym_id"] == sample.record.structure.pocket_chain_id] = 2

        # Compute features
        try:
            features = dataset.featurizer.process(
                    tokenized,
                    training=True,
                    max_atoms=self.max_atoms if self.pad_to_max_atoms else None,
                    max_tokens=self.max_tokens if self.pad_to_max_tokens else None,
                    max_seqs=self.max_seqs,
                    pad_to_max_seqs=self.pad_to_max_seqs,
                    symmetries=self.symmetries,
                    atoms_per_window_queries=self.atoms_per_window_queries,
                    min_dist=self.min_dist,
                    max_dist=self.max_dist,
                    num_bins=self.num_bins,
                    compute_symmetries=self.return_symmetries,
                    binder_pocket_conditioned_prop=self.binder_pocket_conditioned_prop,
                    binder_pocket_cutoff=self.binder_pocket_cutoff,
                    binder_pocket_sampling_geometric_p=self.binder_pocket_sampling_geometric_p,
                )
    
        except Exception as e:
            print(f"Featurizer failed on {sample.record.id} with error {e}. Skipping.")
            return self.__getitem__(idx)

        features["pdb_id"] = torch.tensor([ord(c) for c in sample.record.id])
        features["seq"] = from_numpy(unmask_res_type).long()
        features["cdr_mask"] = from_numpy(cdr_token_mask).bool()
        features["attn_mask"] = torch.ones_like(features["seq"]).bool()
        features["region_type"] = from_numpy(region_type).long()
        features["chain_type"] = chain_type
        if self.max_tokens is not None:
            pad_len = self.max_tokens - len(features["seq"])
            features["seq"] = pad_dim(features["seq"], 0, pad_len)
            features["cdr_mask"] = pad_dim(features["cdr_mask"], 0, pad_len)
            features["chain_type"] = pad_dim(features["chain_type"], 0, pad_len)
            features["region_type"] = pad_dim(features["region_type"], 0, pad_len)
            features["attn_mask"] = pad_dim(features["attn_mask"], 0, pad_len)

        return features

    def __len__(self) -> int:
        """Get the length of the dataset.

        Returns
        -------
        int
            The length of the dataset.

        """
        return self.samples_per_epoch


class ValidationDataset(torch.utils.data.Dataset):
    """Base iterable dataset."""

    def __init__(
        self,
        datasets: list[Dataset],
        seed: int,
        symmetries: dict,
        distinguish_epitope: bool = True,
        max_atoms: Optional[int] = None,
        max_tokens: Optional[int] = None,
        max_seqs: Optional[int] = None,
        pad_to_max_atoms: bool = False,
        pad_to_max_tokens: bool = False,
        pad_to_max_seqs: bool = False,
        atoms_per_window_queries: int = 32,
        min_dist: float = 2.0,
        max_dist: float = 22.0,
        num_bins: int = 64,
        overfit: Optional[int] = None,
        crop_validation: bool = False,
        return_symmetries: Optional[bool] = False,
        binder_pocket_conditioned_prop: Optional[float] = 0.0,
        binder_pocket_cutoff: Optional[float] = 6.0,
    ) -> None:
        """Initialize the validation dataset."""
        super().__init__()
        self.datasets = datasets
        self.distinguish_epitope = distinguish_epitope
        self.max_atoms = max_atoms
        self.max_tokens = max_tokens
        self.max_seqs = max_seqs
        self.seed = seed
        self.symmetries = symmetries
        self.random = np.random if overfit else np.random.RandomState(self.seed)
        self.pad_to_max_tokens = pad_to_max_tokens
        self.pad_to_max_atoms = pad_to_max_atoms
        self.pad_to_max_seqs = pad_to_max_seqs
        self.overfit = overfit
        self.crop_validation = crop_validation
        self.atoms_per_window_queries = atoms_per_window_queries
        self.min_dist = min_dist
        self.max_dist = max_dist
        self.num_bins = num_bins
        self.return_symmetries = return_symmetries
        self.binder_pocket_conditioned_prop = binder_pocket_conditioned_prop
        self.binder_pocket_cutoff = binder_pocket_cutoff

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        """Get an item from the dataset.

        Parameters
        ----------
        idx : int
            The data index.

        Returns
        -------
        dict[str, Tensor]
            The sampled data features.

        """
        # Pick dataset based on idx
        for dataset in self.datasets:
            size = len(dataset.manifest.records)
            if self.overfit is not None:
                size = min(size, self.overfit)
            if idx < size:
                break
            idx -= size

        # Get a sample from the dataset
        record = dataset.manifest.records[idx]

        # Get the structure
        try:
            input_data = load_input(record, dataset.target_dir, dataset.msa_dir, dataset.no_msa)
        except Exception as e:
            print(f"Failed to load input for {record.id} with error {e}. Skipping.")
            return self.__getitem__(0)

        # Tokenize structure
        try:
            tokenized, spec_token_mask = dataset.tokenizer.tokenize(input_data)
        except Exception as e:
            print(f"Tokenizer failed on {record.id} with error {e}. Skipping.")
            return self.__getitem__(idx)

        h_region_type = ab_region_type(tokenized.tokens, spec_token_mask, record.structure.H_chain_id)
        l_region_type = ab_region_type(tokenized.tokens, spec_token_mask, record.structure.L_chain_id)
        ag_region_types = ag_region_type(tokenized.tokens, spec_token_mask, 
                                         [record.structure.H_chain_id, record.structure.L_chain_id],
                                         self.distinguish_epitope)
        region_type = h_region_type + l_region_type + ag_region_types

        # Compute crop
        try:
            if self.crop_validation and (self.max_tokens is not None):
                tokenized, spec_token_mask, region_type = dataset.cropper.crop(
                    tokenized,
                    token_mask=spec_token_mask,
                    token_region=region_type,
                    random=np.random,
                    max_tokens=self.max_tokens,
                    h_chain_id=record.structure.H_chain_id,
                    l_chain_id=record.structure.L_chain_id,
                )
        except Exception as e:
            print(f"Cropper failed on {record.id} with error {e}. Skipping.")
            return self.__getitem__(0)

        # Check if there are tokens
        if len(tokenized.tokens) == 0:
            msg = "No tokens in cropped structure."
            raise ValueError(msg)

        indices = [i for i, x in enumerate(tokenized.tokens) if x["asym_id"] in [record.structure.H_chain_id, record.structure.L_chain_id]]
        cdr_token_mask = np.zeros_like(spec_token_mask, dtype=bool)
        cdr_token_mask[indices] = spec_token_mask[indices]
        
        unmask_res_type = cp.deepcopy(tokenized.tokens["res_type"])
        all_masked_res_type = torch.where(cdr_token_mask, const.unk_token_ids["PROTEIN"], unmask_res_type)
        tokenized.tokens["res_type"] = all_masked_res_type

        chain_type = torch.ones_like(from_numpy(tokenized.tokens["asym_id"])).long() * 3
        chain_type[tokenized.tokens["asym_id"] == record.structure.H_chain_id] = 1
        chain_type[tokenized.tokens["asym_id"] == record.structure.L_chain_id] = 2

        # Compute features
        try:
            pad_atoms = self.crop_validation and self.pad_to_max_atoms
            pad_tokens = self.crop_validation and self.pad_to_max_tokens

            features = dataset.featurizer.process(
                tokenized,
                training=False,
                max_atoms=self.max_atoms if pad_atoms else None,
                max_tokens=self.max_tokens if pad_tokens else None,
                max_seqs=self.max_seqs,
                pad_to_max_seqs=self.pad_to_max_seqs,
                symmetries=self.symmetries,
                atoms_per_window_queries=self.atoms_per_window_queries,
                min_dist=self.min_dist,
                max_dist=self.max_dist,
                num_bins=self.num_bins,
                compute_symmetries=self.return_symmetries,
                binder_pocket_conditioned_prop=self.binder_pocket_conditioned_prop,
                binder_pocket_cutoff=self.binder_pocket_cutoff,
                binder_pocket_sampling_geometric_p=1.0,  # this will only sample a single pocket token
                only_ligand_binder_pocket=True,
            )
        except Exception as e:
            print(f"Featurizer failed on {record.id} with error {e}. Skipping.")
            return self.__getitem__(0)

        features["pdb_id"] = torch.tensor([ord(c) for c in record.id])
        features["seq"] = from_numpy(unmask_res_type).long()
        features["cdr_mask"] = from_numpy(cdr_token_mask).bool()
        features["attn_mask"] = torch.ones_like(features["seq"]).bool()
        features["region_type"] = from_numpy(region_type).long()
        features["chain_type"] = chain_type
        if self.crop_validation and self.pad_to_max_tokens and self.max_tokens is not None:
            pad_len = self.max_tokens - len(features["seq"])
            features["seq"] = pad_dim(features["seq"], 0, pad_len)
            features["cdr_mask"] = pad_dim(features["cdr_mask"], 0, pad_len)
            features["chain_type"] = pad_dim(features["chain_type"], 0, pad_len)
            features["region_type"] = pad_dim(features["region_type"], 0, pad_len)
            features["attn_mask"] = pad_dim(features["attn_mask"], 0, pad_len)

        return features

    def __len__(self) -> int:
        """Get the length of the dataset.

        Returns
        -------
        int
            The length of the dataset.

        """
        if self.overfit is not None:
            length = sum(len(d.manifest.records[: self.overfit]) for d in self.datasets)
        else:
            length = sum(len(d.manifest.records) for d in self.datasets)

        return length


class BoltzTrainingDataModule(pl.LightningDataModule):
    """DataModule for boltz."""

    def __init__(self, cfg: DataConfig) -> None:
        """Initialize the DataModule.

        Parameters
        ----------
        config : DataConfig
            The data configuration.

        """
        super().__init__()
        self.cfg = cfg # load the data config

        assert self.cfg.val_batch_size == 1, "Validation only works with batch size=1."
    
        # Load symmetries
        symmetries = get_symmetries(cfg.symmetries)

        # Load datasets
        train: list[Dataset] = []
        val: list[Dataset] = []

        # iterate over the datasets, for us, there is only one dataset: SabDab
        for data_config in cfg.datasets:
            # Set target_dir
            target_dir = Path(data_config.target_dir) # the path to the target directory
            msa_dir = Path(data_config.msa_dir) # the path to the msa directory

            # Load manifest
            if data_config.manifest_path is not None:
                path = Path(data_config.manifest_path)
            else:
                path = target_dir / "manifest.json"
            manifest: Manifest = Manifest.load(path)

            # Split records if given
            train_records = []
            val_records = []
            if data_config.train_split is not None:
                with open(data_config.train_split, "r") as file:
                    train_split = set(json.load(file))
                for record in manifest.records:
                    if record.id in train_split:
                        train_records.append(record)
            elif data_config.val_split is not None:
                with open(data_config.val_split, "r") as file:
                    val_split = set(json.load(file))
                for record in manifest.records:
                    if record.id in val_split:
                        val_records.append(record)
            else:
                for record in manifest.records:
                    train_records.append(record)

            # Filter training records
            train_records = [
                record
                for record in train_records
                if all(f.filter(record) for f in cfg.filters)
            ]
            # Filter training records
            if data_config.filters is not None:
                train_records = [
                    record
                    for record in train_records
                    if all(f.filter(record) for f in data_config.filters)
                ]
                
            if cfg.no_msa:
                print("[Info] No msa provided")
                
            # Create train dataset
            train_manifest = Manifest(train_records)
            train.append(
                Dataset(
                    target_dir,
                    msa_dir,
                    train_manifest,
                    data_config.prob,
                    data_config.sampler,
                    data_config.cropper,
                    cfg.no_msa,
                    cfg.tokenizer,
                    cfg.featurizer,
                )
            )

            # Create validation dataset
            if val_records:
                val_manifest = Manifest(val_records)
                val.append(
                    Dataset(
                        target_dir,
                        msa_dir,
                        val_manifest,
                        data_config.prob,
                        data_config.sampler,
                        data_config.cropper,
                        cfg.no_msa,
                        cfg.tokenizer,
                        cfg.featurizer,
                    )
                )

        # Print dataset sizes
        for dataset in train:
            dataset: Dataset
            print(f"[Info] Training dataset size: {len(dataset.manifest.records)}")

        for dataset in val:
            dataset: Dataset
            print(f"[Info] Validation dataset size: {len(dataset.manifest.records)}")

        # Create wrapper datasets
        self._train_set = TrainingDataset(
            datasets=train,
            samples_per_epoch=cfg.samples_per_epoch,
            distinguish_epitope=cfg.distinguish_epitope,
            max_atoms=cfg.max_atoms,
            max_tokens=cfg.max_tokens,
            max_seqs=cfg.max_seqs,
            pad_to_max_atoms=cfg.pad_to_max_atoms,
            pad_to_max_tokens=cfg.pad_to_max_tokens,
            pad_to_max_seqs=cfg.pad_to_max_seqs,
            symmetries=symmetries,
            atoms_per_window_queries=cfg.atoms_per_window_queries,
            min_dist=cfg.min_dist,
            max_dist=cfg.max_dist,
            num_bins=cfg.num_bins,
            overfit=cfg.overfit,
            binder_pocket_conditioned_prop=cfg.train_binder_pocket_conditioned_prop,
            binder_pocket_cutoff=cfg.binder_pocket_cutoff,
            binder_pocket_sampling_geometric_p=cfg.binder_pocket_sampling_geometric_p,
            return_symmetries=cfg.return_train_symmetries,
        )
        self._val_set = ValidationDataset(
            datasets=train if cfg.overfit is not None else val,
            seed=cfg.random_seed,
            max_atoms=cfg.max_atoms,
            max_tokens=cfg.max_tokens,
            max_seqs=cfg.max_seqs,
            pad_to_max_atoms=cfg.pad_to_max_atoms,
            pad_to_max_tokens=cfg.pad_to_max_tokens,
            pad_to_max_seqs=cfg.pad_to_max_seqs,
            symmetries=symmetries,
            distinguish_epitope=cfg.distinguish_epitope,
            atoms_per_window_queries=cfg.atoms_per_window_queries,
            min_dist=cfg.min_dist,
            max_dist=cfg.max_dist,
            num_bins=cfg.num_bins,
            overfit=cfg.overfit,
            crop_validation=cfg.crop_validation,
            return_symmetries=cfg.return_val_symmetries,
            binder_pocket_conditioned_prop=cfg.val_binder_pocket_conditioned_prop,
            binder_pocket_cutoff=cfg.binder_pocket_cutoff,
        )

    def setup(self, stage: Optional[str] = None) -> None:
        """Run the setup for the DataModule.

        Parameters
        ----------
        stage : str, optional
            The stage, one of 'fit', 'validate', 'test'.

        """
        return

    def train_dataloader(self) -> DataLoader:
        """Get the training dataloader.

        Returns
        -------
        DataLoader
            The training dataloader.

        """
        return DataLoader(
            self._train_set,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
            shuffle=True,
            collate_fn=collate,
        )

    def val_dataloader(self) -> DataLoader:
        """Get the validation dataloader.

        Returns
        -------
        DataLoader
            The validation dataloader.

        """
        return DataLoader(
            self._val_set,
            batch_size=self.cfg.val_batch_size,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
            shuffle=False,
            collate_fn=collate,
        )

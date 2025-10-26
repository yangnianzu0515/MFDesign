from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
import copy as cp
from typing import Optional
from torch import Tensor, from_numpy
from torch.utils.data import DataLoader
from boltz.data.feature.pad import pad_dim
from boltz.data import const
from boltz.data.feature.featurizer import BoltzFeaturizer
from boltz.data.feature.pad import pad_to_max
from boltz.data.tokenize.boltz import BoltzTokenizer
from boltz.data.types import MSA, Input, Manifest, Record, Structure, AntibodyInfo
from boltz.data.module.training import ab_region_type, ag_region_type

def load_input(record: Record, target_dir: Path, msa_dir: Path) -> Input:
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
    structure = np.load(target_dir / f"{record.id}.npz")
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
    data : List[Dict[str, Tensor]]
        The data to collate.

    Returns
    -------
    Dict[str, Tensor]
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
            "record",
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


class PredictionDataset(torch.utils.data.Dataset):
    """Base iterable dataset."""

    def __init__(
        self,
        manifest: Manifest,
        target_dir: Path,
        msa_dir: Path,
        inpaint: bool = False,
        ground_truth_dir: Optional[Path] = None,
        use_epitope: bool = True
    ) -> None:
        """Initialize the training dataset.

        Parameters
        ----------
        manifest : Manifest
            The manifest to load data from.
        target_dir : Path
            The path to the target directory.
        msa_dir : Path
            The path to the msa directory.

        """
        super().__init__()
        self.manifest = manifest
        self.target_dir = target_dir
        self.msa_dir = msa_dir
        self.tokenizer = BoltzTokenizer()
        self.featurizer = BoltzFeaturizer()
        self.inpaint = inpaint
        self.ground_truth_dir = ground_truth_dir
        self.use_epitope = use_epitope

    def __getitem__(self, idx: int) -> dict:
        """Get an item from the dataset.

        Returns
        -------
        Dict[str, Tensor]
            The sampled data features.

        """
        # Get a sample from the dataset
        record = self.manifest.records[idx]

        # Get the structure
        try:
            input_data = load_input(record, self.target_dir, self.msa_dir)
        except Exception as e:  # noqa: BLE001
            print(f"Failed to load input for {record.id} with error {e}. Skipping.")  # noqa: T201
            return self.__getitem__(0)

        # Tokenize structure
        try:
            tokenized, spec_token_mask = self.tokenizer.tokenize(input_data)
        except Exception as e:  # noqa: BLE001
            print(f"Tokenizer failed on {record.id} with error {e}. Skipping.")  # noqa: T201
            return self.__getitem__(0)
        seq_mask = np.zeros_like(tokenized.tokens["res_type"], dtype=bool)
        if isinstance(record.structure, AntibodyInfo):
            seq_mask[(tokenized.tokens["res_type"] == 22) & 
                     (tokenized.tokens["asym_id"] == record.structure.H_chain_id)] = True
            seq_mask[(tokenized.tokens["res_type"] == 22) & 
                     (tokenized.tokens["asym_id"] == record.structure.L_chain_id)] = True

        if self.inpaint:
            # Load the ground truth
            try:
                ground_truth = np.load(self.ground_truth_dir / f"{record.id}.npz")
                ground_truth = Structure(
                    atoms=ground_truth["atoms"],
                    bonds=ground_truth["bonds"],
                    residues=ground_truth["residues"],
                    chains=ground_truth["chains"],
                    connections=ground_truth["connections"],
                    interfaces=ground_truth["interfaces"],
                    mask=ground_truth["mask"],
                )

                ground_truth_tokens = self.tokenizer.tokenize(Input(ground_truth, {}))[0].tokens[:len(tokenized.tokens)]

                for i, (token, ground_truth_token) in enumerate(zip(tokenized.tokens, ground_truth_tokens)):
                    if spec_token_mask[i]:
                        continue
                
                    assert token["atom_num"] == ground_truth_token["atom_num"]
                    assert token["res_idx"] == ground_truth_token["res_idx"]
                    assert token["res_type"] == ground_truth_token["res_type"]
                    assert token["asym_id"] == ground_truth_token["asym_id"]
                
                coord_data = []
                resolved_mask = []
                coord_mask = []
                for i, token in enumerate(ground_truth_tokens):
                    start = token["atom_idx"]
                    end = token["atom_idx"] + token["atom_num"]
                    token_atoms = ground_truth.atoms[start:end]
                    if len(token_atoms) < tokenized.tokens[i]["atom_num"]:
                        token_atoms = np.concatenate([token_atoms, 
                        np.zeros(tokenized.tokens[i]["atom_num"] - len(token_atoms), dtype=token_atoms.dtype)])
                    coord_data.append(np.array([token_atoms["coords"]]))
                    resolved_mask.append(token_atoms["is_present"])
                    if seq_mask[i]:
                        coord_mask.append(np.ones_like(token_atoms["is_present"], dtype=bool))
                    else:
                        coord_mask.append(1 - token_atoms["is_present"])
                
                resolved_mask = from_numpy(np.concatenate(resolved_mask))
                coord_mask = from_numpy(np.concatenate(coord_mask))
                coords = from_numpy(np.concatenate(coord_data, axis=1))

                assert(len(coord_mask) == len(resolved_mask))
                assert(len(coord_mask) == coords.shape[1])

                center = (coords * resolved_mask[None, :, None]).sum(dim=1)
                center = center / resolved_mask.sum().clamp(min=1)
                coords = coords - center[:, None]

                atoms_per_window_queries = 32
                pad_len = (
                    (len(resolved_mask) - 1) // atoms_per_window_queries + 1
                ) * atoms_per_window_queries - len(resolved_mask)
                coords = pad_dim(coords, 1, pad_len)
                coord_mask = pad_dim(coord_mask, 0, pad_len)
                resolved_mask = pad_dim(resolved_mask, 0, pad_len)
            except Exception as e:
                print(f"Failed to load ground truth for {record.id} with error {e}. Skipping.") 
                return self.__getitem__(0)
        else:
            coords = coord_mask = resolved_mask = None

        if isinstance(record.structure, AntibodyInfo):
            h_region_type = ab_region_type(tokenized.tokens, spec_token_mask, record.structure.H_chain_id)
            l_region_type = ab_region_type(tokenized.tokens, spec_token_mask, record.structure.L_chain_id)
            ag_region_types = ag_region_type(tokenized.tokens, spec_token_mask, [record.structure.H_chain_id, record.structure.L_chain_id], self.use_epitope)
            region_type = h_region_type + l_region_type + ag_region_types
            
        assert len(region_type) == len(spec_token_mask)
        
        # Inference specific options
        options = record.inference_options
        if options is None:
            binders, pocket = None, None
        else:
            binders, pocket = options.binders, options.pocket

        if isinstance(record.structure, AntibodyInfo):
            indices = [i for i, x in enumerate(tokenized.tokens) if x["asym_id"] in [record.structure.H_chain_id, record.structure.L_chain_id]]
            cdr_token_mask = np.zeros_like(spec_token_mask, dtype=bool)
            cdr_token_mask[indices] = spec_token_mask[indices]
            chain_type = torch.ones_like(from_numpy(tokenized.tokens["asym_id"])).long() * 3
            chain_type[tokenized.tokens["asym_id"] == record.structure.H_chain_id] = 1
            chain_type[tokenized.tokens["asym_id"] == record.structure.L_chain_id] = 2

        # Compute features
        try:
            features = self.featurizer.process(
                tokenized,
                training=False,
                max_atoms=None,
                max_tokens=None,
                max_seqs=const.max_msa_seqs,
                pad_to_max_seqs=False,
                symmetries={},
                compute_symmetries=False,
                inference_binder=binders,
                inference_pocket=pocket,
            )
        except Exception as e:  # noqa: BLE001
            print(f"Featurizer failed on {record.id} with error {e}. Skipping.")  # noqa: T201
            return self.__getitem__(0)

        if coords is not None:
            features["coords_gt"] = coords
            features["coord_mask"] = coord_mask
            assert features["atom_resolved_mask"].shape == resolved_mask.shape
            features["atom_resolved_mask"] = resolved_mask
            assert features["coords"].shape == features["coords_gt"].shape

        features["record"] = record
        features["masked_seq"] = from_numpy(cp.deepcopy(tokenized.tokens["res_type"])).long()
        features["pdb_id"] = torch.tensor([ord(c) for c in record.id])
        features["seq_mask"] = from_numpy(seq_mask).bool()
        features["cdr_mask"] = from_numpy(cdr_token_mask).bool()
        features["attn_mask"] = torch.ones_like(features["cdr_mask"]).bool()
        features["region_type"] = from_numpy(region_type).long()
        features["chain_type"] = chain_type

        return features

    def __len__(self) -> int:
        """Get the length of the dataset.

        Returns
        -------
        int
            The length of the dataset.

        """
        return len(self.manifest.records)


class BoltzInferenceDataModule(pl.LightningDataModule):
    """DataModule for Boltz inference."""

    def __init__(
        self,
        manifest: Manifest,
        target_dir: Path,
        msa_dir: Path,
        num_workers: int,
        inpaint: bool = False,
        ground_truth_dir: Optional[Path] = None,
        use_epitope: bool = True
    ) -> None:
        """Initialize the DataModule.

        Parameters
        ----------
        config : DataConfig
            The data configuration.

        """
        super().__init__()
        self.num_workers = num_workers
        self.manifest = manifest
        self.target_dir = target_dir
        self.msa_dir = msa_dir
        self.inpaint = inpaint
        self.ground_truth_dir = ground_truth_dir
        self.use_epitope = use_epitope

    def predict_dataloader(self) -> DataLoader:
        """Get the training dataloader.

        Returns
        -------
        DataLoader
            The training dataloader.

        """
        dataset = PredictionDataset(
            manifest=self.manifest,
            target_dir=self.target_dir,
            msa_dir=self.msa_dir,
            inpaint=self.inpaint,
            ground_truth_dir=self.ground_truth_dir,
            use_epitope=self.use_epitope
        )
        return DataLoader(
            dataset,
            batch_size=1,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,
            collate_fn=collate,
        )

    def transfer_batch_to_device(
        self,
        batch: dict,
        device: torch.device,
        dataloader_idx: int,  # noqa: ARG002
    ) -> dict:
        """Transfer a batch to the given device.

        Parameters
        ----------
        batch : Dict
            The batch to transfer.
        device : torch.device
            The device to transfer to.
        dataloader_idx : int
            The dataloader index.

        Returns
        -------
        np.Any
            The transferred batch.

        """
        for key in batch:
            if key not in [
                "all_coords",
                "all_resolved_mask",
                "crop_to_all_atom_map",
                "chain_symmetries",
                "amino_acids_symmetries",
                "ligand_symmetries",
                "record",
            ]:
                batch[key] = batch[key].to(device)
        return batch

import argparse
import json
import multiprocessing
import pickle
import traceback

import sys
sys.path.insert(0, '/mnt/nas-new/home/yangnianzu/icml/boltz/src/')

from dataclasses import asdict, dataclass, replace
from functools import partial
from pathlib import Path
from typing import Any, Optional
import yaml
import numpy as np
import rdkit
import pandas as pd
from mmcif import parse_mmcif
from redis import Redis
from tqdm import tqdm
from boltz.data.filter.static.filter import StaticFilter
from boltz.data.filter.static.ligand import ExcludedLigands
from boltz.data.filter.static.polymer import (
    ClashingChainsFilter,
    ConsecutiveCA,
    MinimumLengthFilter,
    UnknownFilter,
)

from boltz.data.types import ChainInfo, InterfaceInfo, Record, Target
from boltz.data import const

import os
import subprocess
from multiprocessing import Pool
from joblib import Parallel, delayed
from ruamel.yaml import YAML

@dataclass(frozen=True, slots=True)
class PDB:
    """A raw MMCIF PDB file."""

    id: str
    path: str
    cif_path: str
    yaml_path: str
    is_scfv: bool
    resolution: float
    heavy_id: str
    light_id: str
    antigen_id: list
    select_seq_dict: dict
    masked_seq_dict: dict
    cdr_select: str
    cdr_chain_select_list: list

class Resource:
    """A shared resource for processing."""

    def __init__(self, host: str, port: int) -> None:
        """Initialize the redis database."""
        self._redis = Redis(host=host, port=port)

    def get(self, key: str) -> Any:  # noqa: ANN401
        """Get an item from the Redis database."""
        value = self._redis.get(key)
        if value is not None:
            value = pickle.loads(value)  # noqa: S301
        return value

    def __getitem__(self, key: str) -> Any:  # noqa: ANN401
        """Get an item from the resource."""
        out = self.get(key)
        if out is None:
            raise KeyError(key)
        return out

def parse_yaml_entity(path: Path) -> dict:
    with path.open("r") as file:
        schema = yaml.load(file, Loader=yaml.CLoader) 

    version = schema.get("version", 1)
    if version != 1:
        msg = f"Invalid version {version} in input!"
        raise ValueError(msg)

    items_to_group = {}
    for item in schema["sequences"]:
        entity_type = next(iter(item.keys())).lower()
        if entity_type not in {"protein", "dna", "rna", "ligand"}:
            msg = f"Invalid entity type: {entity_type}"
            raise ValueError(msg)

        if entity_type in {"protein", "dna", "rna"}:
            seq = str(item[entity_type]["sequence"])
        items_to_group.setdefault((entity_type, seq), []).append(item)

    chains: dict[str, int] = {}
    for entity_id, items in enumerate(items_to_group.values()):
        entity_type = next(iter(items[0].keys())).lower()

        for item in items:
            ids = item[entity_type]["id"]
            if isinstance(ids, str):
                ids = [ids]
            for chain_name in ids:
                chains[chain_name] = entity_id
    
    return chains


def fetch(
    datadir: Path,
    info_csv_fpath: Path, 
    info_cif_fpath: Path,
    info_yaml_fpath: Path,
    cdr_select: None,
    max_file_size: Optional[int] = None, 
) -> list[PDB]:
    """Fetch the PDB files."""
    with open(info_csv_fpath, 'r') as f:
        info_dict = pd.read_csv(f, index_col=0).to_dict(orient='index')
    
    # Here is the temp path for the cif file.
    # Only for get the metadata. (if not exist, the cif path is None)
    if info_cif_fpath.exists():
        cif_dir_fpath = info_cif_fpath
    else:
        cif_dir_fpath = None
        
    data = []
    excluded = 0
    for key, values in info_dict.items():
        seq_dict = {}
        mask_seq_dict = {}
        split_key = key.split('_')
        pdb_id = split_key[0].lower()

        heavy_id = values['H_chain_id']
        light_id = values['L_chain_id']
        antigen_id = eval(values['antigen_chain_id'])
        
        # Consider about the raw cif: scfv antibody is one sequence.
        is_scfv = values['scfv']
        resolution = values['resolution']
        
        if not pd.isna(heavy_id):
            seq_dict[heavy_id] = values['H_chain_seq'] 
        if not pd.isna(light_id):
            seq_dict[light_id] = values['L_chain_seq']
        if isinstance(antigen_id, list):
            antigen_seq = eval(values['antigen_seq']) 
            for ag_id in antigen_id:
                seq_dict[ag_id] = antigen_seq[ag_id]
        
        
        mask_seq_dict[heavy_id] = values['H_chain_masked_seq']
        mask_seq_dict[light_id] = values['L_chain_masked_seq']
        
        # file = os.path.join(datadir, f'{pdb_id}.cif')      
        file = os.path.join(datadir, f'{pdb_id}.pdb')      
        
        if cif_dir_fpath is not None:
            cif_file = f'{pdb_id}.cif'
            cif_file = os.path.join(cif_dir_fpath, cif_file)
        else:
            cif_file = None
        
        if info_yaml_fpath is not None:
            yaml_file = os.path.join(info_yaml_fpath, f'{key}.yaml')
        else:  
            yaml_file = None

        # Check file size and skip if too large
        if max_file_size is not None and (file.stat().st_size > max_file_size):
            excluded += 1
            continue
        
        # Get the corresponding chain name:
        true_chain_name_list = []
        if cdr_select is not None and 'H' in cdr_select:
            true_chain_name_list.append(heavy_id)
        elif cdr_select is not None and 'L' in cdr_select: 
            true_chain_name_list.append(light_id)

        # Create the target
        target = PDB(
            id=pdb_id, 
            path=str(file),
            cif_path=cif_file,
            yaml_path=yaml_file,
            is_scfv=is_scfv,
            resolution=resolution,
            heavy_id=heavy_id,
            light_id=light_id,
            antigen_id=antigen_id,
            select_seq_dict=seq_dict,
            masked_seq_dict=mask_seq_dict,
            cdr_select=cdr_select,
            cdr_chain_select_list=true_chain_name_list
        )
        data.append(target)

    print(f"Excluded {excluded} files due to size.")  # noqa: T201
    return data

def finalize(outdir: Path) -> None:
    """Run post-processing in main thread.

    Parameters
    ----------
    outdir : Path
        The output directory.

    """
    # Group records into a manifest
    records_dir = outdir / "records"

    failed_count = 0
    records = []
    for record in records_dir.iterdir():
        path = records_dir / record
        try:
            with path.open("r") as f:
                records.append(json.load(f))
        except:
            failed_count += 1
            print(f"Failed to parse {record}")
    print(f"Failed to parse {failed_count} entries)")

    print(f"Processed {len(records)} entries.")

    # Save manifest
    outpath = outdir / "manifest.json"
    with outpath.open("w") as f:
        json.dump(records, f)


def parse(data: PDB, resource: Resource) -> Target:
    """Process a structure.

    Parameters
    ----------
    data : PDB
        The raw input data.
    resource: Resource
        The shared resource.

    Returns
    -------
    Target
        The processed data.

    """
    # Get the PDB id
    pdb_id = data.id.lower()

    # Parse structure
    parsed = parse_mmcif(data, resource)
    structure = parsed.data
    structure_info = parsed.info

    # Create chain metadata
    chain_info = []
    antigen_id_str = ''.join(data.antigen_id) if isinstance(data.antigen_id, list) else ''
    antibody_heavy_id = data.heavy_id if not pd.isna(data.heavy_id) else ''
    antibody_light_id = data.light_id if not pd.isna(data.light_id) else ''
    chain_names = antibody_heavy_id + antibody_light_id + antigen_id_str
    id = f"{pdb_id}_{antibody_heavy_id}_{antibody_light_id}_{antigen_id_str}"
    
    yaml_entity = parse_yaml_entity(Path(data.yaml_path))

    for i, chain in enumerate(structure.chains):
        key = f"{pdb_id}_{chain['entity_id']}"
        assert chain['name'] == chain_names[i]
        entity_id = yaml_entity[chain['name']]
        msa_id = f"{id}_{entity_id}"
        chain_info.append(
            ChainInfo(
                chain_id=i,
                chain_name=chain["name"],
                msa_id=msa_id,  # FIX. we modified as -1.
                mol_type=int(chain["mol_type"]),
                cluster_id=-1,
                num_residues=int(chain["res_num"]),
            )
        )

    # Get interface metadata
    interface_info = []
    for interface in structure.interfaces:
        chain_1 = int(interface["chain_1"])
        chain_2 = int(interface["chain_2"])
        interface_info.append(
            InterfaceInfo(
                chain_1=chain_1,
                chain_2=chain_2,
            )
        )

    # Create record
    record = Record(
        id=id,
        structure=structure_info,
        chains=chain_info,
        interfaces=interface_info,
    )

    return Target(structure=structure, record=record)

def process_structure(
    data,
    args,
    filters: list[StaticFilter],
) -> None:
    """Process a target.

    Parameters
    ----------
    data : PDB
        The raw input data.
    resource: Resource
        The shared resource.
    outdir : Path
        The output directory.

    """
    outdir = args.outdir

    # Load filters
    filters = [
        ExcludedLigands(),
        MinimumLengthFilter(min_len=4, max_len=5000),
        UnknownFilter(),
        ConsecutiveCA(max_dist=10.0),
        ClashingChainsFilter(freq=0.3, dist=1.7),
    ]
    resource = Resource(host=args.redis_host, port=args.redis_port)
    
    # Check if we need to process
    # add data antigen_id for same protein with different antigen.
    antigen_id_str = ''.join(data.antigen_id) if isinstance(data.antigen_id, list) else ''
    antibody_heavy_id = data.heavy_id if not pd.isna(data.heavy_id) else ''
    antibody_light_id = data.light_id if not pd.isna(data.light_id) else ''
    struct_path = outdir / "structures" / f"{data.id}_{antibody_heavy_id}_{antibody_light_id}_{antigen_id_str}.npz"
    record_path = outdir / "records" / f"{data.id}_{antibody_heavy_id}_{antibody_light_id}_{antigen_id_str}.json"
    pdb_id_name = f'{data.id}_{antibody_heavy_id}_{antibody_light_id}_{antigen_id_str}'

    try:
        # Parse the target
        target: Target = parse(data, resource)
        structure = target.structure

        # Apply the filters
        mask = structure.mask
        if filters is not None:
            for f in filters:
                if f.__class__.__name__ == 'ConsecutiveCA':
                    filter_mask = f.filter(structure, target.record.structure.H_chain_id, target.record.structure.L_chain_id)
                else:    
                    filter_mask = f.filter(structure)
                if filter_mask.sum() != len(mask):
                    print(f"Filter {f.__class__.__name__} failed for {data.id}")
                    return pdb_id_name  # tmp_name
                mask = mask & filter_mask
    except Exception as e:  # noqa: BLE001
        print(f"Failed to parse {data.id}, error: {e}")
        return pdb_id_name  # tmp_name

    # Replace chains and interfaces
    chains = []
    for i, chain in enumerate(target.record.chains):
        chains.append(replace(chain, valid=bool(mask[i])))

    interfaces = []
    for interface in target.record.interfaces:
        chain_1 = bool(mask[interface.chain_1])
        chain_2 = bool(mask[interface.chain_2])
        interfaces.append(replace(interface, valid=(chain_1 and chain_2)))

    # Replace structure and record
    structure = replace(structure, mask=mask)
    record = replace(target.record, chains=chains, interfaces=interfaces)
    target = replace(target, structure=structure, record=record)
    
    # Dump structure
    np.savez_compressed(struct_path, **asdict(structure))

    # Dump record
    with record_path.open("w") as f:
        json.dump(asdict(record), f)
    
    return None


def process(args) -> None:
    """Run the data processing task."""
    # Create output directory
    args.outdir.mkdir(parents=True, exist_ok=True)

    # Create output directories
    records_dir = args.outdir / "records"
    records_dir.mkdir(parents=True, exist_ok=True)

    structure_dir = args.outdir / "structures"
    structure_dir.mkdir(parents=True, exist_ok=True)

    # Load filters
    filters = [
        ExcludedLigands(),
        MinimumLengthFilter(min_len=4, max_len=5000),
        UnknownFilter(),
        ConsecutiveCA(max_dist=10.0),
        ClashingChainsFilter(freq=0.3, dist=1.7),
    ]

    # Set default pickle properties
    pickle_option = rdkit.Chem.PropertyPickleOptions.AllProps
    rdkit.Chem.SetDefaultPickleProperties(pickle_option)

    # Check if we can run in parallel
    num_processes = min(args.num_processes, multiprocessing.cpu_count())
    # num_processes = 2
    parallel = num_processes > 1

    # Get data points
    print("Fetching data...")
    data = fetch(
                args.datadir, 
                args.processed_csv_fpath, 
                args.cif_path,
                args.yaml_path,
                args.cdr_select,
            )

    # Randomly permute the data
    random = np.random.RandomState()
    permute = random.permutation(len(data))
    data = [data[i] for i in permute]

    # Run processing
    if parallel:
        # Create processing function
        Parallel(n_jobs=num_processes)(
                delayed(process_structure)(
                    item, args, filters
                ) for item in tqdm(
                    data, 
                    total=len(data),
                    desc="Processing data",
                    dynamic_ncols=True
                )
            )
        
    else:
        for item in tqdm(data, total=len(data)):
            # Here for debug. we set the temp for select the specific pdb.
            if item.id != '7sr8':
                continue
            process_structure(
                item,
                args,
                filters=filters
            )

    # Finalize
    finalize(args.outdir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process structure data.")
    parser.add_argument(
        "--datadir",
        type=Path,
        default='./data/raw_data/chothia',
        help="The data containing the PDB files.",
    )
    parser.add_argument(
        "--processed_csv_fpath",
        type=Path,
        default="./data/summary-decuplication-distance_threshold_9.csv",
        help="Path to the preprocessed file.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default="./data/antibody_data",
        help="The output directory.",
    )
    parser.add_argument(
        "--cif_path",
        type=Path,
        default="./data/raw_data/cif",
        help="The cif file path.",
    )
    parser.add_argument(
        "--yaml_path",
        type=Path,
        default="./data/yaml_for_data_after_decuplication-distance_threshold_9",
        help="The yaml file path.",
    )
    parser.add_argument(
        "--num-processes",
        type=int,
        default=multiprocessing.cpu_count(),
        help="The number of processes.",
    )
    parser.add_argument(
        "--redis-host",
        type=str,
        default="localhost",
        help="The Redis host.",
    )
    parser.add_argument(
        "--redis-port",
        type=int,
        default=7777,
        help="The Redis port.",
    )
    parser.add_argument(
        "--cdr_select",
        default=None, # H3, L3, only support one region selection now.
        help="The cdr region to select.",
    )
    args = parser.parse_args()
    process(args)


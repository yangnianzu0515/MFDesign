import pickle
import sys
sys.path.insert(0, './src')
import os
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Optional
import shutil
import click
import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.utilities import rank_zero_only
from tqdm import tqdm
import json


from boltz.data import const
from boltz.data.module.inference import BoltzInferenceDataModule
from boltz.data.msa.mmseqs2 import run_mmseqs2
from boltz.data.parse.a3m import parse_a3m
from boltz.data.parse.csv import parse_csv_for_ab_design, parse_csv
from boltz.data.parse.fasta import parse_fasta
from boltz.data.parse.yaml import parse_yaml
from boltz.data.types import MSA, Manifest, Record
from boltz.data.write.writer import BoltzWriter
from boltz.model.model import Boltz1
import argparse 

CCD_URL = "https://huggingface.co/boltz-community/boltz-1/resolve/main/ccd.pkl"
MODEL_URL = (
    "https://huggingface.co/boltz-community/boltz-1/resolve/main/boltz1_conf.ckpt"
)


@dataclass
class BoltzProcessedInput:
    """Processed input data."""

    manifest: Manifest
    targets_dir: Path
    msa_dir: Path


@dataclass
class BoltzDiffusionParams:
    """Diffusion process parameters."""

    gamma_0: float = 0.605
    gamma_min: float = 1.107
    noise_scale: float = 0.901
    rho: float = 8
    step_scale: float = 1.638
    temperature: float = 1.0
    sigma_min: float = 0.0004
    sigma_max: float = 160.0
    sigma_data: float = 16.0
    P_mean: float = -1.2
    P_std: float = 1.5
    coordinate_augmentation: bool = True
    alignment_reverse_diff: bool = True
    synchronize_sigmas: bool = True
    use_inference_model_cache: bool = True
    noise_type: str = "discrete_absorb"
    
@dataclass
class AF3DiffusionParams:
    """Diffusion process parameters."""

    gamma_0: float = 0.8
    gamma_min: float = 1.0
    noise_scale: float = 1.0
    rho: float = 7
    step_scale: float = 1.0
    temperature: float = 1.0
    sigma_min: float = 0.0004
    sigma_max: float = 160.0
    sigma_data: float = 16.0
    P_mean: float = -1.2
    P_std: float = 1.5
    coordinate_augmentation: bool = True
    alignment_reverse_diff: bool = True
    synchronize_sigmas: bool = True
    use_inference_model_cache: bool = True
    noise_type: str = "discrete_absorb"



@rank_zero_only
def download(cache: Path) -> None:
    """Download all the required data.

    Parameters
    ----------
    cache : Path
        The cache directory.

    """
    # Download CCD
    ccd = cache / "ccd.pkl"
    if not ccd.exists():
        click.echo(
            f"Downloading the CCD dictionary to {ccd}. You may "
            "change the cache directory with the --cache flag."
        )
        urllib.request.urlretrieve(CCD_URL, str(ccd))  # noqa: S310

    # Download model
    model = cache / "boltz1_conf.ckpt"
    if not model.exists():
        click.echo(
            f"Downloading the model weights to {model}. You may "
            "change the cache directory with the --cache flag."
        )
        urllib.request.urlretrieve(MODEL_URL, str(model))  # noqa: S310


def check_inputs(
    data: Path,
    outdir: Path,
    override: bool = False,
) -> list[Path]:
    """Check the input data and output directory.

    If the input data is a directory, it will be expanded
    to all files in this directory. Then, we check if there
    are any existing predictions and remove them from the
    list of input data, unless the override flag is set.

    Parameters
    ----------
    data : Path
        The input data.
    outdir : Path
        The output directory.
    override: bool
        Whether to override existing predictions.

    Returns
    -------
    list[Path]
        The list of input data.

    """
    click.echo("Checking input data.")

    # Check if data is a directory
    if data.is_dir(): # if the input data is a directory, expand the directory to all files in this directory
        data: list[Path] = list(data.glob("*")) # data.glob("*") returns a list of all files in the directory, each file is a Path object

        # Filter out non .fasta or .yaml files, raise
        # an error on directory and other file types
        filtered_data = [] # filtered_data is a list of data with allowed file types
        for d in data:
            if d.suffix in (".fa", ".fas", ".fasta", ".yml", ".yaml"):
                filtered_data.append(d)
            elif d.is_dir():
                msg = f"Found directory {d} instead of .fasta or .yaml."
                raise RuntimeError(msg)
            else:
                msg = (
                    f"Unable to parse filetype {d.suffix}, "
                    "please provide a .fasta or .yaml file."
                )
                raise RuntimeError(msg)

        data = filtered_data
    else:
        data = [data]

    # Check if existing predictions are found
    existing = (outdir / "predictions").rglob("*")
    existing = {e.name for e in existing if e.is_dir()} # existing is a set of the names of directories in the predictions directory

    # Remove them from the input data
    if existing and not override: 
    # if there are existing predictions and the override flag is not set to True, we will skip the input data that has already been predicted
        data = [d for d in data if d.stem not in existing] 
        # remove the existing predictions from the input data
        num_skipped = len(existing) - len(data) 
        # num_skipped is the number of existing predictions that will be skipped
        msg = (
            f"Found some existing predictions ({num_skipped}), "
            f"skipping and running only the missing ones, "
            "if any. If you wish to override these existing "
            "predictions, please set the --override flag."
        )
        click.echo(msg)
    elif existing and override: 
        # override is set to True, we will override the existing predictions
        msg = "Found existing predictions, will override."
        click.echo(msg)

    return data


def compute_msa(
    data: dict[str, str],
    target_id: str,
    msa_dir: Path,
    msa_server_url: str,
    msa_pairing_strategy: str,
) -> None:
    """Compute the MSA for the input data.

    Parameters
    ----------
    data : dict[str, str]
        The input protein sequences.
    target_id : str
        The target id.
    msa_dir : Path
        The msa directory.
    msa_server_url : str
        The MSA server URL.
    msa_pairing_strategy : str
        The MSA pairing strategy.

    """
    
    # in our ab design, len(data) is always greater than 1 (antibody + antigen)
    # First, we need to obtain the paired MSA
    if len(data) > 1:
        paired_msas = run_mmseqs2(
            list(data.values()),
            msa_dir / f"{target_id}_paired_tmp",
            use_env=True,
            use_pairing=True,
            host_url=msa_server_url,
            pairing_strategy=msa_pairing_strategy,
        )
        """
        The format of a3mlines is like this:
        [
            ">101\nseq1\n>UniRef100_Q9Y6K8\nseq2\n>UniRef100_222228\nseq3\n",
            ">102\nseq1\n>UniRef100_123328\nseq2\n>UniRef100_22SDS3\nseq3\n",
            ">103\nseq1\n>UniRef100_9234K2\nseq2\n>UniRef100_22FSF2\nseq3\n",
            ...
        ]
        
        seq may be "DUMMY"
        """
    else:
        paired_msas = [""] * len(data)


    unpaired_msa = run_mmseqs2(
        list(data.values()),
        msa_dir / f"{target_id}_unpaired_tmp",
        use_env=True,
        use_pairing=False,
        host_url=msa_server_url,
        pairing_strategy=msa_pairing_strategy,
    )

    for idx, name in enumerate(data):
        # Get paired sequences
        paired = paired_msas[idx].strip().splitlines()
        paired = paired[1::2]  # ignore headers
        # paired = paired[: const.max_paired_seqs]
        """
        const.max_msa_seqs = 16384 (paired + unpaired)
        const.max_paired_seqs = 8192 (paired)
        """        
        

        # Set key per row and remove empty sequences
        keys = [idx for idx, s in enumerate(paired) if s != "-" * len(s)]
        # keys is a list of indices of the paired sequences that are not empty (DUMMY sequences are not included)
        paired = [s for s in paired if s != "-" * len(s)]
        # paired is a list of the paired sequences that are not empty (DUMMY sequences are not included)

        # Combine paired-unpaired sequences
        unpaired = unpaired_msa[idx].strip().splitlines()
        unpaired = unpaired[1::2]
        # unpaired = unpaired[: (const.max_msa_seqs - len(paired))]
        # unpaired is a list of the unpaired sequences
        if paired:
            unpaired = unpaired[1:]  # ignore query is already present
            # unpaired[1:] is the sequence itself that we are interested in

        # Combine
        seqs = paired + unpaired
        keys = keys + [-1] * len(unpaired)
        # keys = -1 means the sequence is the results of unpaired msa

        # Dump MSA
        csv_str = ["key,sequence"] + [f"{key},{seq}" for key, seq in zip(keys, seqs)]

        msa_path = msa_dir / f"{name}.csv"
        # the format of name is like this: target_id_entity_id
        # where target_id is the file name of the yaml file in our case
        
        with msa_path.open("w") as f:
            f.write("\n".join(csv_str))


@rank_zero_only
def process_inputs(  # noqa: C901, PLR0912, PLR0915
    data: list[Path],
    out_dir: Path,
    ccd_path: Path,
    msa_dir: Path,
    preprocessed_data_path: str,
    msa_filtering_threshold: float,
    max_msa_seqs: int = 4096,
    processed_msa_dir: Optional[Path] = None,
    use_msa_server: bool = False,
    msa_server_url: str = None,
    msa_pairing_strategy: str = None,
    only_process_msa: bool = False,
    generate_msa: bool = False
) -> dict:
    """Process the input data and output directory.

    Parameters
    ----------
    data : list[Path]
        The input data.
    out_dir : Path
        The output directory.
    ccd_path : Path
        The path to the CCD dictionary.
    max_msa_seqs : int, optional
        Max number of MSA sequences, by default 4096.
    use_msa_server : bool, optional
        Whether to use the MMSeqs2 server for MSA generation, by default False.

    Returns
    -------
    BoltzProcessedInput
        The processed input data.

    """
    click.echo("Processing input data.")

    # Create output directories
    structure_dir = out_dir / "processed" / "structures"
    predictions_dir = out_dir / "predictions"

    out_dir.mkdir(parents=True, exist_ok=True)
    msa_dir.mkdir(parents=True, exist_ok=True)
    structure_dir.mkdir(parents=True, exist_ok=True)
    processed_msa_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    # Load CCD
    with ccd_path.open("rb") as file:
        ccd = pickle.load(file)  # noqa: S301
        
    # load preprocessed data
    with open(preprocessed_data_path, "r") as file:
        preprocessed_data = json.load(file)

    # Parse input data
    records: list[Record] = []
    chain_infos = {}
    for path in tqdm(data):
        # Parse data
        if path.suffix in (".fa", ".fas", ".fasta"):
            target = parse_fasta(path, ccd)
        elif path.suffix in (".yml", ".yaml"):
            target, now_chain_infos = parse_yaml(path, ccd)
            chain_infos[target.record.id] = now_chain_infos
        elif path.is_dir():
            msg = f"Found directory {path} instead of .fasta or .yaml, skipping."
            raise RuntimeError(msg)
        else:
            msg = (
                f"Unable to parse filetype {path.suffix}, "
                "please provide a .fasta or .yaml file."
            )
            raise RuntimeError(msg)

        # Get target id
        target_id = target.record.id
        # target_id the file name of the yaml file in our case

        # Get all MSA ids and decide whether to generate MSA
        to_generate = {}
        prot_id = const.chain_type_ids["PROTEIN"]
        # prot_id is the id of the protein chain type, i.e. 0, in our ab design, chains are always protein chains, prot_id is always 0
        
        for chain in target.record.chains:
            # Add to generate list, assigning entity id
            if (chain.mol_type == prot_id) and (chain.msa_id == 0):
                # msa_id==0 means we have not generated the msa for this chain yet
                # else, msa_id is the precomputed msa file path
                entity_id = chain.entity_id
                # obtain the entity id of the chain. Note that may be several chains with the same entity id if their sequences are the same
                
                msa_id = f"{target_id}_{entity_id}"
                # target_id is the file name of the yaml file in our case
                # entity_id is the entity id of the chain
                gt_seq = chain_infos[target.record.id]["entity_to_gt"][entity_id]
                # Use ground truth sequence to compute MSA for better quality, since it has more coevolutionary coupling when pairing
                to_generate[msa_id] = gt_seq if gt_seq else target.sequences[entity_id]
                chain.msa_id = msa_dir / f"{msa_id}.csv"

            # We do not support msa generation for non-protein chains
            elif chain.msa_id == 0:
                # this branch means the chain is not a protein chain, we will not encounter this branch in our ab design
                chain.msa_id = -1

        # to_generate is not empty means we have not generated the msa for some chains; we need to check set the use_msa_server flag to True
        # Generate MSA
        if to_generate and not use_msa_server and generate_msa:
            msg = "Missing MSA's in input and --use_msa_server flag not set."
            raise RuntimeError(msg)

        # # to_generate is not empty means we have not generated the msa for some chains; then we need to generate the msa for these chains
        if to_generate and generate_msa:
            msg = f"Generating MSA for {path} with {len(to_generate)} protein entities."
            click.echo(msg)
            compute_msa(
                data=to_generate,
                target_id=target_id,
                msa_dir=msa_dir,
                msa_server_url=msa_server_url,
                msa_pairing_strategy=msa_pairing_strategy,
            )

        # Parse MSA data
        msas = sorted({c.msa_id for c in target.record.chains if c.msa_id != -1})
        # msas is a list of the msa paths and sorted by the entity id
                
        msa_id_map = {}
        for msa_idx, msa_id in enumerate(msas):
            # Check that raw MSA exists
            msa_path = Path(msa_id)
            # Dump processed MSA
            processed = processed_msa_dir / f"{target_id}_{msa_idx}.npz"
            msa_id_map[msa_id] = f"{target_id}_{msa_idx}"

            # check if the msa file exists, if not, raise an error
            if (not processed.exists()) and (not msa_path.exists()):
                msg = f"Processed MSA file {processed} not found."
                raise FileNotFoundError(msg)
            # check if the processed msa file exists, if not, we need to parse the msa file
            if not processed.exists():
                # in our ab design, the msa file is always a csv file
                if msa_path.suffix == ".csv":
                    if msa_filtering_threshold is not None:
                        msa: MSA = parse_csv_for_ab_design(
                            msa_path, max_seqs=max_msa_seqs, 
                            entry_info=preprocessed_data[target_id], 
                            msa_filtering_threshold=msa_filtering_threshold
                        )[0]
                    else:
                        msa: MSA = parse_csv(msa_path, max_seqs=max_msa_seqs)
                else:
                    msg = f"MSA file {msa_path} not supported, only csv."
                    raise RuntimeError(msg)

                msa.dump(processed)

        if only_process_msa:
            continue

        for c in target.record.chains:
            if (c.msa_id != -1) and (c.msa_id in msa_id_map):
                c.msa_id = msa_id_map[c.msa_id]

        # Keep record
        records.append(target.record)

        # Dump structure
        struct_path = structure_dir / f"{target.record.id}.npz"
        target.structure.dump(struct_path)

    if only_process_msa:
        return

    # Dump manifest
    manifest = Manifest(records)
    manifest.dump(out_dir / "processed" / "manifest.json")

    if len(chain_infos) > 0:
        with open(out_dir / "processed" / "chain_infos.json", "w") as f:
            json.dump(chain_infos, f)

def check_checkpoint(pretrained: Path, checkpoint: Path) -> None: 

    pretrained_dict = torch.load(pretrained, map_location="cpu")
    checkpoint_dict = torch.load(checkpoint, map_location="cpu")

    is_change = False
    # Remove the unused parameters
    for k, v in checkpoint_dict["hyper_parameters"]["score_model_args"]["sequence_model_args"].items():
        if k not in {"hidden_dim", "vocab_size", "dropout"}:
            is_change = True
            del checkpoint_dict["hyper_parameters"]["score_model_args"]["sequence_model_args"][k]
    
    # If use self trained checkpoint, it is necessary to manually add confidence_module for inference.
    module_params = {
        k: v for k, v in pretrained_dict["state_dict"].items() 
        if k.startswith('confidence_module') or k.startswith('structure_module.out_token_feat_update')
    }
    for k,v in module_params.items():
        if k not in checkpoint_dict["state_dict"]:
            is_change = True
            checkpoint_dict["state_dict"][k] = v
    
    if is_change:
        click.echo("The checkpoint has been changed, we will save it.")
        torch.save(checkpoint_dict, checkpoint)


def predict(
    data: str,
    out_dir: str,
    preprocessed_data_path: str,
    msa_dir: Optional[str] = None,
    processed_msa_dir: Optional[str] = None,
    structure_inpainting: bool = False,
    ground_truth_structure_dir: Optional[str] = None,
    cache: str = "~/.boltz",
    checkpoint: Optional[str] = None,
    devices: int = 1,
    accelerator: str = "gpu",
    recycling_steps: int = 3,
    sampling_steps: int = 200,
    diffusion_samples: int = 1,
    step_scale: float = 1.638,
    temperature: float = 1.0,
    write_full_pae: bool = False,
    write_full_pde: bool = False,
    output_format: Literal["pdb", "mmcif"] = "mmcif",
    num_workers: int = 4,
    override: bool = False,
    seed: Optional[int] = None,
    use_msa_server: bool = False,
    msa_server_url: str = "https://api.colabfold.com",
    msa_pairing_strategy: str = "greedy",
    only_process_msa: bool = False,
    msa_filtering_threshold: float = 0.2,
    noise_type: str = "discrete_absorb",
    sequence_prediction: bool = True,
    use_epitope: bool = True
) -> None:
    """Run predictions with Boltz-1."""
    # If cpu, write a friendly warning
    if accelerator == "cpu":
        msg = "Running on CPU, this will be slow. Consider using a GPU."
        click.echo(msg)

    if structure_inpainting and ground_truth_structure_dir is None:
        click.echo("Please provide the ground truth structure directory if inpainting.")
        return

    if ground_truth_structure_dir is not None:
        ground_truth_structure_dir = Path(ground_truth_structure_dir).expanduser()

    # Set no grad
    torch.set_grad_enabled(False) 
    # disable gradient computation for inference

    # Ignore matmul precision warning
    torch.set_float32_matmul_precision("highest") 
    # set the precision of the matmul operation to highest

    # Set seed if desired
    if seed is not None:
        seed_everything(seed)

    # Set cache path
    cache = Path(cache).expanduser() 
    # expand the cache path like ~/.boltz to the absolute path like /home/yangnianzu/.boltz
    cache.mkdir(parents=True, exist_ok=True) 
    # create the cache directory if it does not exist and all parent directories will be created too

    # Create output directories
    data = Path(data).expanduser()
    out_dir = Path(out_dir).expanduser()
    out_dir = out_dir / f"boltz_results_{data.stem}" 
    # data.stem is the name of the input data without the extension and without the father directory
    out_dir.mkdir(parents=True, exist_ok=True)

    is_generate = False
    if msa_dir is None and processed_msa_dir is None:
        is_generate = True
        click.echo("No MSA directory provided.")
    
    if msa_dir is None:
        msa_dir = out_dir / "msa"
    else:
        msa_dir = Path(msa_dir).expanduser()

    if processed_msa_dir is not None:
        processed_msa_dir = Path(processed_msa_dir).expanduser()
    else:
        processed_msa_dir = out_dir / "processed" / "msa"

    # Download necessary data and model
    download(cache) 
    # if the local cache is not found, download the data and model from the remote server

    # Validate inputs
    data = check_inputs(data, out_dir, override) 
    # return the data files in the input data directory (may skip some files if we have already predicted them, depends on the override flag)
    if not data: 
        # if the input data is empty, we will exit
        click.echo("No predictions to run, exiting.")
        return

    # Set up trainer
    strategy = "auto"
    if (isinstance(devices, int) and devices > 1) or (
        isinstance(devices, list) and len(devices) > 1
    ):
        strategy = DDPStrategy()
        if len(data) < devices:
            msg = (
                "Number of requested devices is greater "
                "than the number of predictions."
            )
            raise ValueError(msg)

    msg = f"Running predictions for {len(data)} structure"
    msg += "s" if len(data) > 1 else ""
    click.echo(msg)

    # Process inputs
    ccd_path = cache / "ccd.pkl"
    process_inputs(
        data=data,
        out_dir=out_dir,
        msa_dir=msa_dir,
        processed_msa_dir=processed_msa_dir,
        preprocessed_data_path=preprocessed_data_path,
        ccd_path=ccd_path,
        use_msa_server=use_msa_server,
        msa_server_url=msa_server_url,
        msa_pairing_strategy=msa_pairing_strategy,
        msa_filtering_threshold=msa_filtering_threshold,
        only_process_msa=only_process_msa,
        generate_msa=is_generate,
    )

    if only_process_msa:
        click.echo("We have only precomputed the MSA for the input data and exit.")
        return

    # Load processed data
    processed_dir = out_dir / "processed"
    processed = BoltzProcessedInput(
        manifest=Manifest.load(processed_dir / "manifest.json"),
        targets_dir=processed_dir / "structures",
        msa_dir=processed_msa_dir,
    )

    # Create data module
    data_module = BoltzInferenceDataModule(
        manifest=processed.manifest,
        target_dir=processed.targets_dir,
        msa_dir=processed.msa_dir,
        num_workers=num_workers,
        inpaint=structure_inpainting,
        ground_truth_dir=ground_truth_structure_dir,
        use_epitope=use_epitope
    )

    pretrained = cache / "boltz1_conf.ckpt"
    # Load model
    if checkpoint is None:
        checkpoint = pretrained
    else:
        checkpoint = Path(checkpoint).expanduser()

    predict_args = {
        "recycling_steps": recycling_steps,
        "sampling_steps": sampling_steps,
        "diffusion_samples": diffusion_samples,
        "write_confidence_summary": True,
        "write_full_pae": write_full_pae,
        "write_full_pde": write_full_pde,
    }
    # diffusion_params = AF3DiffusionParams()
    diffusion_params = BoltzDiffusionParams()
    diffusion_params.step_scale = step_scale
    diffusion_params.temperature = temperature
    diffusion_params.noise_type = noise_type

    if checkpoint != pretrained:
        check_checkpoint(pretrained, checkpoint)

    model_module: Boltz1 = Boltz1.load_from_checkpoint(
        checkpoint,
        strict=False,
        predict_args=predict_args,
        structure_prediction_training=False,
        sequence_prediction_training=sequence_prediction,
        confidence_prediction=True,
        confidence_imitate_trunk=True,
        structure_inpainting=structure_inpainting,
        alpha_pae=1.0,
        map_location="cpu",
        diffusion_process_args=asdict(diffusion_params),
        ema=False,
    )
    click.echo(f"Loaded model from checkpoint {checkpoint}")
    model_module.eval()

    # Create prediction writer
    pred_writer = BoltzWriter(
        data_dir=processed.targets_dir,
        output_dir=out_dir / "predictions",
        output_format=output_format,
        seq_info_path=processed_dir / "chain_infos.json",
    )

    trainer = Trainer(
        default_root_dir=out_dir,
        strategy=strategy,
        callbacks=[pred_writer],
        accelerator=accelerator,
        devices=devices,
        precision=32,
    )

    # Compute predictions
    trainer.predict(
        model_module,
        datamodule=data_module,
        return_predictions=False,
    )

def run():
    parser = argparse.ArgumentParser(description="Run Boltz predictions directly.")
    parser.add_argument("--data", type=str, required=True, help="Path to input data.")
    parser.add_argument(
        "--out_dir", type=str, default="./", help="Path to save predictions."
    )
    parser.add_argument(
        "--cache", type=str, default="./model", help="Path to cache directory."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./model/stage_4.ckpt",
        help="Optional path to checkpoint file.",
    )
    parser.add_argument("--devices", type=int, default=1, help="Number of devices.")
    parser.add_argument(
        "--accelerator",
        choices=["gpu", "cpu", "tpu"],
        default="gpu",
        help="Compute accelerator.",
    )
    parser.add_argument("--recycling_steps", type=int, default=3, help="Recycling steps.")
    parser.add_argument("--sampling_steps", type=int, default=200, help="Sampling steps.")
    parser.add_argument(
        "--diffusion_samples", type=int, default=1, help="Diffusion samples."
    )
    parser.add_argument(
        "--step_scale", type=float, default=1.638, help="Step scale parameter."
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="Temperature parameter."
    )
    parser.add_argument(
        "--write_full_pae", action="store_true", help="Write full PAE to file."
    )
    parser.add_argument(
        "--write_full_pde", action="store_true", help="Write full PDE to file."
    )
    parser.add_argument(
        "--output_format",
        choices=["pdb", "mmcif"],
        default="pdb",
        help="Output format.",
    )
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers.")
    parser.add_argument(
        "--override", action="store_true", help="Override existing predictions."
    )
    parser.add_argument("--seed", type=int, default=2025, help="Random seed.")
    parser.add_argument(
        "--use_msa_server",
        action="store_true",
        help="Use MMSeqs2 server for MSA generation.",
    )
    parser.add_argument(
        "--msa_server_url",
        type=str,
        default="https://api.colabfold.com",
        help="MSA server URL.",
    )
    parser.add_argument(
        "--msa_pairing_strategy",
        type=str,
        default="greedy",
        help="MSA pairing strategy.",
    )
    
    # only precompute the MSA for the input data
    parser.add_argument(
        "--only_process_msa",
        action="store_true",
        help="Only precompute the MSA for the input data.",
    )
    
    # threshold for the MSA filtering
    parser.add_argument(
        "--msa_filtering_threshold",
        type=float,
        default=0.2,
        help="The threshold for the MSA masking.",
    )

    parser.add_argument(
        "--preprocessed_data_path",
        type=str,
        default="./data/summary.json",
        help="Path to preprocessed data.",
    )
    
    parser.add_argument(
        "--msa_dir",
        type=str,
        default=None,
        help="Path to msa data(csv).",
    )

    parser.add_argument(
        "--processed_msa_dir",
        type=str,
        default=None,
        help="Path to processed msa data.",
    )

    parser.add_argument(
        "--structure_inpainting",
        action="store_true",
        help="Whether to perform structure inpainting.",
    )

    parser.add_argument(
        "--ground_truth_structure_dir",
        type=str,
        default="./data/antibody_data/structures",
        help="Path to ground truth structure data.",
    )

    parser.add_argument(
        "--noise_type",
        type=str,
        choices=['discrete_absorb', 'discrete_uniform', 'continuous'],
        default="discrete_absorb",
        help="Noise type.",
    )
    
    parser.add_argument(
        "--only_structure_prediction",
        action="store_false",
        help="Only predict structure, no sequence prediction result and no sequence model used",
    )
    
    parser.add_argument(
        "--no_epitope",
        action="store_false",
        help="Not support epitope region",
    )

    args = parser.parse_args()
    predict(
        data=args.data,
        out_dir=args.out_dir,
        cache=args.cache,
        checkpoint=args.checkpoint,
        devices=args.devices,
        accelerator=args.accelerator,
        recycling_steps=args.recycling_steps,
        sampling_steps=args.sampling_steps,
        diffusion_samples=args.diffusion_samples,
        step_scale=args.step_scale,
        temperature=args.temperature,
        write_full_pae=args.write_full_pae,
        write_full_pde=args.write_full_pde,
        output_format=args.output_format,
        num_workers=args.num_workers,
        override=args.override,
        seed=args.seed,
        use_msa_server=args.use_msa_server,
        msa_server_url=args.msa_server_url,
        msa_pairing_strategy=args.msa_pairing_strategy,
        only_process_msa=args.only_process_msa,
        msa_filtering_threshold=args.msa_filtering_threshold,
        msa_dir=args.msa_dir,
        processed_msa_dir=args.processed_msa_dir,
        preprocessed_data_path=args.preprocessed_data_path,
        ground_truth_structure_dir=args.ground_truth_structure_dir,
        structure_inpainting=args.structure_inpainting,
        noise_type=args.noise_type,
        sequence_prediction=args.only_structure_prediction,
        use_epitope=args.no_epitope
    )

if __name__ == "__main__":
    run()

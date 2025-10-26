from dataclasses import asdict, replace
import json
from pathlib import Path
from typing import Literal, Optional

import numpy as np
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import BasePredictionWriter
import torch
from torch import Tensor

from boltz.data.types import (
    Interface,
    Record,
    Structure,
)
from boltz.data import const
from boltz.data.write.mmcif import to_mmcif
from boltz.data.write.pdb import to_pdb

def calculate_aar(seq_predict, seq_truth, seq_mask):
    assert len(seq_predict) == len(seq_truth) == len(seq_mask)
    
    segments = []
    start = -1
    for i in range(len(seq_mask)):
        if seq_mask[i] == '1' and start == -1:
            start = i
        elif seq_mask[i] == '0' and start != -1:
            segments.append((start, i - 1))
            start = -1
    if start != -1:
        segments.append((start, len(seq_mask) - 1))

    accuracies = []
    total_matches = 0
    total_count = 0
    
    for (start, end) in segments:
        seg_len = end - start + 1
        matches = sum(1 for i in range(start, end + 1) if seq_predict[i] == seq_truth[i])
        accuracies.append(matches / seg_len)
        
        total_matches += matches
        total_count += seg_len
    
    total_accuracy = total_matches / total_count
    
    h_acc_segments = segments[:3]
    h_acc_matches = sum(
        sum(1 for i in range(start, end + 1) if seq_predict[i] == seq_truth[i])
        for (start, end) in h_acc_segments
    )
    h_acc_len = sum(end - start + 1 for (start, end) in h_acc_segments)
    h_acc = h_acc_matches / h_acc_len if h_acc_len > 0 else 0
    
    l_acc_segments = segments[-3:]
    l_acc_matches = sum(
        sum(1 for i in range(start, end + 1) if seq_predict[i] == seq_truth[i])
        for (start, end) in l_acc_segments
    )
    l_acc_len = sum(end - start + 1 for (start, end) in l_acc_segments)
    l_acc = l_acc_matches / l_acc_len if l_acc_len > 0 else 0
    
    return accuracies, total_accuracy, h_acc, l_acc

class BoltzWriter(BasePredictionWriter):
    """Custom writer for predictions."""

    def __init__(
        self,
        data_dir: str,
        output_dir: str,
        output_format: Literal["pdb", "mmcif"] = "mmcif",
        seq_info_path: Optional[str] = None,
    ) -> None:
        """Initialize the writer.

        Parameters
        ----------
        output_dir : str
            The directory to save the predictions.

        """
        super().__init__(write_interval="batch")
        if output_format not in ["pdb", "mmcif"]:
            msg = f"Invalid output format: {output_format}"
            raise ValueError(msg)

        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_format = output_format
        self.failed = 0
        self.seq_info_path = seq_info_path

        # Create the output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write_on_batch_end(
        self,
        trainer: Trainer,  # noqa: ARG002
        pl_module: LightningModule,  # noqa: ARG002
        prediction: dict[str, Tensor],
        batch_indices: list[int],  # noqa: ARG002
        batch: dict[str, Tensor],
        batch_idx: int,  # noqa: ARG002
        dataloader_idx: int,  # noqa: ARG002
    ) -> None:
        """Write the predictions to disk."""
        if prediction["exception"]:
            self.failed += 1
            return

        # Get the records
        records: list[Record] = batch["record"]

        # Get the predictions
        coords = prediction["coords"]
        coords = coords.unsqueeze(0)

        pad_masks = prediction["masks"]

        seqs = prediction["seqs"]
        
        if seqs is not None:
            seqs = seqs.unsqueeze(0)
            assert seqs.shape[0] == coords.shape[0]
        else:
            seqs = [""] * len(records)

        # Get ranking
        argsort = torch.argsort(prediction["confidence_score"], descending=True)
        idx_to_rank = {idx.item(): rank for rank, idx in enumerate(argsort)}

        seqs_info = {}
        if self.seq_info_path is not None:
            seqs_info = json.load(open(self.seq_info_path))

        # Iterate over the records
        for record, coord, pad_mask, seq in zip(records, coords, pad_masks, seqs):
            # Load the structure
            path = self.data_dir / f"{record.id}.npz"
            structure: Structure = Structure.load(path)

            # Compute chain map with masked removed, to be used later
            chain_map = {}
            for i, mask in enumerate(structure.mask):
                if mask:
                    chain_map[len(chain_map)] = i

            # Remove masked chains completely
            structure = structure.remove_invalid_chains()

            # Save the structure
            struct_dir = self.output_dir / record.id
            struct_dir.mkdir(exist_ok=True)

            if len(seq) > 0:
                seq_info = seqs_info.get(record.id, None)
                seq_path = struct_dir / f"{record.id}.seq"
                with seq_path.open("w") as f:
                    seq_gt = seq_info["seq_gt"] if seq_info is not None else None
                    spec_mask = seq_info["spec_mask"] if seq_info is not None else None
                    title_str = "Rank\tSequence\tTotal\tH\tL\tH1\tH2\tH3\tL1\tL2\tL3\n" if seq_gt else "Rank\tSequence\n"
                    f.write(title_str)
                    lines = {}
                    for model_idx in range(seq.shape[0]):
                        seq_str = "".join([const.prot_token_to_letter[const.tokens[int(x.item())]] for x in seq[model_idx]])
                        if seq_gt:
                            seq_str = seq_str[:len(seq_gt)] if seq_gt else seq_str
                            accuracies, total_accuracy, h_acc, l_acc = calculate_aar(seq_str, seq_gt, spec_mask)
                            aar_str = "\t".join([f"{acc:.3f}" for acc in accuracies])
                            lines[idx_to_rank[model_idx]] = f"{idx_to_rank[model_idx]}\t{seq_str}\t{total_accuracy:.3f}\t{h_acc:.3f}\t{l_acc:.3f}\t{aar_str}\n"
                        else:
                            lines[idx_to_rank[model_idx]] = f"{idx_to_rank[model_idx]}\t{seq_str}\n"
                        
                    sorted_lines = {k: lines[k] for k in sorted(lines)}
                    for line in sorted_lines.values():
                        f.write(line)

            for model_idx in range(coord.shape[0]):
                # Get model coord
                model_coord = coord[model_idx]
                # Unpad
                coord_unpad = model_coord[pad_mask.bool()]
                coord_unpad = coord_unpad.cpu().numpy()

                # New atom table
                atoms = structure.atoms
                atoms["coords"] = coord_unpad
                atoms["is_present"] = True

                # Mew residue table
                residues = structure.residues
                residues["is_present"] = True
                
                if len(seq) > 0:
                    residues["res_type"] = seq[model_idx].cpu().numpy()
                    res_name = [const.tokens[int(x.item())] for x in seq[model_idx]]
                    residues["name"] = np.array(res_name, dtype=np.dtype("<U5"))

                # Update the structure
                interfaces = np.array([], dtype=Interface)
                new_structure: Structure = replace(
                    structure,
                    atoms=atoms,
                    residues=residues,
                    interfaces=interfaces,
                )

                # Update chain info
                chain_info = []
                for chain in new_structure.chains:
                    old_chain_idx = chain_map[chain["asym_id"]]
                    old_chain_info = record.chains[old_chain_idx]
                    new_chain_info = replace(
                        old_chain_info,
                        chain_id=int(chain["asym_id"]),
                        valid=True,
                    )
                    chain_info.append(new_chain_info)

                if self.output_format == "pdb":
                    path = (
                        struct_dir / f"{record.id}_model_{idx_to_rank[model_idx]}.pdb"
                    )
                    with path.open("w") as f:
                        f.write(to_pdb(new_structure))
                elif self.output_format == "mmcif":
                    path = (
                        struct_dir / f"{record.id}_model_{idx_to_rank[model_idx]}.cif"
                    )
                    with path.open("w") as f:
                        if "plddt" in prediction:
                            f.write(
                                to_mmcif(new_structure, prediction["plddt"][model_idx])
                            )
                        else:
                            f.write(to_mmcif(new_structure))
                else:
                    path = (
                        struct_dir / f"{record.id}_model_{idx_to_rank[model_idx]}.npz"
                    )
                    np.savez_compressed(path, **asdict(new_structure))

                # Save confidence summary
                if "plddt" in prediction:
                    path = (
                        struct_dir
                        / f"confidence_{record.id}_model_{idx_to_rank[model_idx]}.json"
                    )
                    confidence_summary_dict = {}
                    for key in [
                        "confidence_score",
                        "ptm",
                        "iptm",
                        "ligand_iptm",
                        "protein_iptm",
                        "complex_plddt",
                        "complex_iplddt",
                        "complex_pde",
                        "complex_ipde",
                    ]:
                        confidence_summary_dict[key] = prediction[key][model_idx].item()
                    confidence_summary_dict["chains_ptm"] = {
                        idx: prediction["pair_chains_iptm"][idx][idx][model_idx].item()
                        for idx in prediction["pair_chains_iptm"]
                    }
                    confidence_summary_dict["pair_chains_iptm"] = {
                        idx1: {
                            idx2: prediction["pair_chains_iptm"][idx1][idx2][
                                model_idx
                            ].item()
                            for idx2 in prediction["pair_chains_iptm"][idx1]
                        }
                        for idx1 in prediction["pair_chains_iptm"]
                    }
                    with path.open("w") as f:
                        f.write(
                            json.dumps(
                                confidence_summary_dict,
                                indent=4,
                            )
                        )

                    # Save plddt
                    plddt = prediction["plddt"][model_idx]
                    path = (
                        struct_dir
                        / f"plddt_{record.id}_model_{idx_to_rank[model_idx]}.npz"
                    )
                    np.savez_compressed(path, plddt=plddt.cpu().numpy())

                # Save pae
                if "pae" in prediction:
                    pae = prediction["pae"][model_idx]
                    path = (
                        struct_dir
                        / f"pae_{record.id}_model_{idx_to_rank[model_idx]}.npz"
                    )
                    np.savez_compressed(path, pae=pae.cpu().numpy())

                # Save pde
                if "pde" in prediction:
                    pde = prediction["pde"][model_idx]
                    path = (
                        struct_dir
                        / f"pde_{record.id}_model_{idx_to_rank[model_idx]}.npz"
                    )
                    np.savez_compressed(path, pde=pde.cpu().numpy())

    def on_predict_epoch_end(
        self,
        trainer: Trainer,  # noqa: ARG002
        pl_module: LightningModule,  # noqa: ARG002
    ) -> None:
        """Print the number of failed examples."""
        # Print number of failed examples
        print(f"Number of failed examples: {self.failed}")  # noqa: T201

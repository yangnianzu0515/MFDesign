from pathlib import Path
from typing import Optional, Tuple
from typing import List
import numpy as np
import pandas as pd

from boltz.data import const
from boltz.data.types import MSA, MSADeletion, MSAResidue, MSASequence

def get_cdr_indices(ref_masked_seq: str) -> List[int]:
    """
    Get the indices of the CDRs of the input sequence
    """
    # find all X positions
    x_indices = [i for i, char in enumerate(ref_masked_seq) if char == 'X']
    
    # find breakpoints (the positions where the distance between two Xs is greater than 1)
    breaks = []
    for i in range(1, len(x_indices)):
        if x_indices[i] - x_indices[i-1] > 1:
            breaks.append(i)
    
    # get the start and end indices for each CDR region
    cdr_indices = [
        x_indices[0], x_indices[breaks[0]-1] + 1,  # CDR1
        x_indices[breaks[0]], x_indices[breaks[1]-1] + 1,  # CDR2
        x_indices[breaks[1]], x_indices[-1] + 1  # CDR3
    ]
    
    return cdr_indices



def filter_msa_df(input_df: pd.DataFrame, query_seq: str, ref_masked_seq: str, msa_filtering_threshold: float) -> pd.DataFrame:
    """
    This function is used to:
    filter out the sequences which share high sequence identity with the CDRs of the first entry of the input_df (which is the query sequence)
    
    Finally, the function will return a new dataframe with the filtered sequences
    """
    input_df.at[0, "sequence"] = ref_masked_seq # replace the query sequence with the reference masked sequence
    
    reserved_entry_indices = [0]
    
    CDR_indices = get_cdr_indices(ref_masked_seq)
    
    query_cdr1 = query_seq[CDR_indices[0]:CDR_indices[1]]
    query_cdr2 = query_seq[CDR_indices[2]:CDR_indices[3]]
    query_cdr3 = query_seq[CDR_indices[4]:CDR_indices[5]]
    
    
    
    def calculate_similarity(seq1: str, seq2: str) -> float:
        """Calculate the similarity between two sequences"""
        matches = sum(a == b for a, b in zip(seq1, seq2))
        return float(matches) / len(seq1)
    
    for idx, entry in enumerate(input_df.iloc[1:].iterrows(), start=1): # notice that the first entry is the query sequence, so we start from the second entry, i.e. index 1
        entry_seq = ''.join([c for c in entry[1]["sequence"] if not c.islower()]) # remove the lower case characters in the sequence
        entry_cdr1 = entry_seq[CDR_indices[0]:CDR_indices[1]]
        entry_cdr2 = entry_seq[CDR_indices[2]:CDR_indices[3]]
        entry_cdr3 = entry_seq[CDR_indices[4]:CDR_indices[5]]   
        
        if calculate_similarity(query_cdr1, entry_cdr1) >= msa_filtering_threshold or \
            calculate_similarity(query_cdr2, entry_cdr2) >= msa_filtering_threshold or \
            calculate_similarity(query_cdr3, entry_cdr3) >= msa_filtering_threshold:
            continue
        else:
            reserved_entry_indices.append(idx)

    return input_df.iloc[reserved_entry_indices]

def parse_csv(
    path: Path,
    max_seqs: Optional[int] = None,
) -> MSA:
    """Process an A3M file.

    Parameters
    ----------
    path : Path
        The path to the a3m(.gz) file.
    max_seqs : int, optional
        The maximum number of sequences.

    Returns
    -------
    MSA
        The MSA object.

    """
    # Read file
    data = pd.read_csv(path)

    # Check columns
    if tuple(sorted(data.columns)) != ("key", "sequence"):
        msg = "Invalid CSV format, expected columns: ['sequence', 'key']"
        raise ValueError(msg)

    # Create taxonomy mapping
    visited = set()
    sequences = []
    deletions = []
    residues = []

    seq_idx = 0
    for line, key in zip(data["sequence"], data["key"]):
        line: str
        line = line.strip()  # noqa: PLW2901
        if not line:
            continue

        # Get taxonomy, if annotated
        taxonomy_id = -1
        if (str(key) != "nan") and (key is not None) and (key != ""):
            taxonomy_id = key

        # Skip if duplicate sequence
        str_seq = line.replace("-", "").upper()
        if str_seq not in visited:
            visited.add(str_seq)
        else:
            continue

        # Process sequence
        residue = []
        deletion = []
        count = 0
        res_idx = 0
        for c in line:
            if c != "-" and c.islower():
                count += 1
                continue
            token = const.prot_letter_to_token[c]
            token = const.token_ids[token]
            residue.append(token)
            if count > 0:
                deletion.append((res_idx, count))
                count = 0
            res_idx += 1

        res_start = len(residues)
        res_end = res_start + len(residue)

        del_start = len(deletions)
        del_end = del_start + len(deletion)

        sequences.append((seq_idx, taxonomy_id, res_start, res_end, del_start, del_end))
        residues.extend(residue)
        deletions.extend(deletion)

        seq_idx += 1
        if (max_seqs is not None) and (seq_idx >= max_seqs):
            break

    # Create MSA object
    msa = MSA(
        residues=np.array(residues, dtype=MSAResidue),
        deletions=np.array(deletions, dtype=MSADeletion),
        sequences=np.array(sequences, dtype=MSASequence),
    )
    return msa



def parse_csv_for_ab_design(
    path: Path,
    entry_info: dict,
    max_seqs: Optional[int] = None,
    msa_filtering_threshold: float = 0.2,
) -> MSA:
    """Process an A3M file.

    Parameters
    ----------
    path : Path
        The path to the a3m(.gz) file.
    max_seqs : int, optional
        The maximum number of sequences.

    Returns
    -------
    MSA
        The MSA object.

    """
    # Read file
    data = pd.read_csv(path)
    num_before_filtering = len(data)
    # Check columns
    if tuple(sorted(data.columns)) != ("key", "sequence"):
        msg = "Invalid CSV format, expected columns: ['sequence', 'key']"
        raise ValueError(msg)

    # Check whether we need to conduct filtering
    file_name = path.stem
    entity_id = file_name.split("_")[-1]
    if entity_id == "0": # it must be the heavy chain
        data = filter_msa_df(data, entry_info["H_chain_seq"], entry_info["H_chain_masked_seq"], msa_filtering_threshold)
    elif entity_id == "1" and not entry_info["L_chain_id"] == None:
        data = filter_msa_df(data, entry_info["L_chain_seq"], entry_info["L_chain_masked_seq"], msa_filtering_threshold)

    num_after_filtering = len(data)

    # Create taxonomy mapping
    visited = set()
    sequences = []
    deletions = []
    residues = []

    seq_idx = 0
    for line, key in zip(data["sequence"], data["key"]):
        line: str
        line = line.strip()  # noqa: PLW2901
        if not line:
            continue

        # Get taxonomy, if annotated
        taxonomy_id = -1
        if (str(key) != "nan") and (key is not None) and (key != ""):
            taxonomy_id = key

        # Skip if duplicate sequence
        str_seq = line.replace("-", "").upper()
        if str_seq not in visited:
            visited.add(str_seq)
        else:
            continue

        # Process sequence
        residue = []
        deletion = []
        count = 0
        res_idx = 0
        for c in line:
            if c != "-" and c.islower():
                # there may exists lower character in the sequence, which is a error
                count += 1
                continue
            token = const.prot_letter_to_token[c]
            # token is a string like "ALA"
            
            token = const.token_ids[token]
            # turn the token into a int id
            
            residue.append(token)
            # ’-‘ is also thought as a normal residue token
            if count > 0:
            # if count > 0, means there exists a deletion right before the current position
                deletion.append((res_idx, count))
                count = 0
            res_idx += 1

        res_start = len(residues)
        # res_start is the start index of the current sequence in the residues array
        res_end = res_start + len(residue)
        # res_end is the end index of the current sequence in the residues array
        # note that residue only contains valid tokens (valid residues and '-')
        del_start = len(deletions)
        del_end = del_start + len(deletion)
        # index from del_start to del_end is the deletion position info corresponding to the current sequence
        sequences.append((seq_idx, taxonomy_id, res_start, res_end, del_start, del_end))
        residues.extend(residue)
        deletions.extend(deletion)

        seq_idx += 1
        if (max_seqs is not None) and (seq_idx >= max_seqs):
            break

    # Create MSA object
    msa = MSA(
        residues=np.array(residues, dtype=MSAResidue),
        deletions=np.array(deletions, dtype=MSADeletion),
        sequences=np.array(sequences, dtype=MSASequence),
    )
    return msa, num_before_filtering, num_after_filtering
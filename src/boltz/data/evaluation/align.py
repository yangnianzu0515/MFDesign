# Align structure through differents methods.
# Direction: CDR align, framework align, whole structure.
from itertools import chain
import numpy as np
import os
import json
import csv
import pandas as pd
import glob

from abnumber import Chain
from Bio import PDB, SeqIO
from Bio.PDB.Polypeptide import index_to_one

from af3_alignment import local_rmsd_from_coords as af3_rmsd_local
from af3_alignment import global_rmsd_from_coords as af3_rmsd_global
from dymean_alignment import compute_rmsd as dymean_rmsd
from utils import get_subdirectories

ONLY_C_ALPHA = {"CA"}
BACKBONE_ATOMS = {"N", "CA", "C", "O"}
UNK_BACKBONE_ATOMS = {"N", "CA", "C", "O", "CB"}

# Get a cdr index for pymol to compute the rmsd.
def cdr_region(chain):
    """input chain (abnumber object)"""
    fr1_length = len(chain.fr1_seq)
    cdr1_length  = len(chain.cdr1_seq)
    fr2_length = len(chain.fr2_seq)
    cdr2_length  = len(chain.cdr2_seq)
    fr3_length = len(chain.fr3_seq)
    cdr3_length  = len(chain.cdr3_seq)
    fr4_length = len(chain.fr4_seq)
    assert sum([fr1_length, cdr1_length, fr2_length, cdr2_length,
           fr3_length, cdr3_length, fr4_length]) == len(chain.seq)
    return (fr1_length, cdr1_length, fr2_length, cdr2_length,
           fr3_length, cdr3_length, fr4_length)


def get_cdr_index(chain):
    
    fr1, cdr1, fr2, cdr2, fr3, cdr3, fr4 = cdr_region(chain)
    cdr1_index = range(fr1+1, fr1+cdr1+1)  # range 左闭右开，需要多加1.
    cdr2_start = fr1 + cdr1 + fr2 
    cdr2_index = range(cdr2_start+1, cdr2_start+cdr2+1)
    cdr3_start = fr1 + cdr1 + fr2 + cdr2 + fr3 
    cdr3_index = range(cdr3_start+1, cdr3_start+cdr3+1)

    return cdr1_index, cdr2_index, cdr3_index


def find_x_intervals_as_ranges(sequence):
    """
    Finds all intervals of consecutive 'X' in the sequence and stores them as range objects.
    The indices in the range objects are 1-based.

    Args:
        sequence (str): The input sequence.

    Returns:
        list: A list of range objects, where each range represents a region containing consecutive 'X'.
    """
    intervals = []
    start = None

    for i, char in enumerate(sequence):
        if char == 'X':
            if start is None:  # Start of a new 'X' region
                start = i + 1  # Convert to 1-based index
        else:
            if start is not None:  # End of an 'X' region
                intervals.append(list(range(start, i + 1)))  # Convert to 1-based range
                start = None

    # Handle the case where the sequence ends with 'X'
    if start is not None:
        intervals.append(list(range(start, len(sequence) + 1)))

    num_X = sequence.count('X')
    len_x = sum(len(l) for l in intervals)
    assert num_X == len_x, 'X indices has problem'

    return intervals


def get_cdr_list(
    seq,  
    numbering_scheme='chothia'
):
    seq_chain = Chain(seq, scheme=numbering_scheme)
    var_seq = seq_chain.seq
    
    cdr1_list, cdr2_list, cdr3_list = get_cdr_index(seq_chain)
    
    return cdr1_list, cdr2_list, cdr3_list, var_seq


def extract_chain_sequence_coord(
    file_path,
    chain_id_list=['H', 'L'],
    coord_type='backbone'
):
    """
    Extract amino acid sequences from each chain in a PDB or CIF file.

    Parameters:
    pdb_file (str): Path to the PDB or CIF file.
    coord_type: CA, backbone, unkbackbone, all 
    Returns:
    dict: A dictionary where keys are chain identifiers and values are amino acid sequences.
    """
    # Parse the structure using PDBParser
    if file_path.endswith(".cif") or file_path.endswith(".mmcif"):
        structure = PDB.MMCIFParser(QUIET=True).get_structure("structure", file_path)
    elif file_path.endswith(".pdb"):
        structure = PDB.PDBParser(QUIET=True).get_structure("structure", file_path)
    else:
        raise ValueError("Unsupported file format. Please provide a .pdb, .cif, or .mmcif file.")

    # sequences = {
    #     chain.id: "".join(
    #         index_to_one(PDB.Polypeptide.three_to_index(residue.resname))
    #         for residue in chain if PDB.is_aa(residue, standard=True)
    #     )
    #     for model in structure for chain in model if chain.id in chain_id_list
    # }
    
    sequences = {}
    coordinates = {}

    for chain in structure.get_chains():
        if chain.id not in chain_id_list:
            continue
            # Extract sequence for the chain
        sequence = []
        chain_coords = {}
        idx = 1   # for after rmsd idx.
        for residue in chain:
            # Residue index
            res_index = idx

            # Convert residue name to single-letter code
            if not PDB.is_aa(residue, standard=False):
                if residue.resname == 'UNK':
                    aa = 'X'
                else:
                    continue
            else:
                try:
                    aa = index_to_one(PDB.Polypeptide.three_to_index(residue.resname))
                except:
                    aa = 'Z'  # if not standard, all of them regard as Z.
            sequence.append(aa)

            # Extract atom coordinates for the residue
            if coord_type == "backbone":
                atom_coords = [
                    atom.coord.tolist() for atom in residue if atom.name in BACKBONE_ATOMS
                ]
            elif coord_type == "unkbackbone":
                atom_coords = [
                    atom.coord.tolist() for atom in residue if atom.name in UNK_BACKBONE_ATOMS
                ]
            elif coord_type == "CA":
                atom_coords = [
                    atom.coord.tolist() for atom in residue if atom.name in ONLY_C_ALPHA
                ]
            elif coord_type == "all":
                atom_coords = [atom.coord.tolist() for atom in residue]
            else:
                raise ValueError("coord_type must be either 'backbone' or 'all'.")
            chain_coords[res_index] = atom_coords
            idx += 1

        # Store the results for the chain
        sequences[chain.id] = "".join(sequence)
        coordinates[chain.id] = chain_coords
    return sequences, coordinates

def calculate_rmsd(
    file_1,
    file_2,
    raw_chain_type_list=['H', 'L'],
    gen_chain_type_list=['A', 'B'],
    calculate_region="CDR",  # Options: "CDR", "FR", "ALL"
    coord_type='backbone',
    rmsd_type='af3_local',   # Options: 'af3_local', 'af3_global'
    verbose=True,
    masked_ratio=1.0,
    backup_seq_dict=None
):
    """
    Calculate RMSD for specific regions (CDR or FR) between two PDB structures.

    Parameters:
    file_1 (str): Path to the first PDB file.
    file_2 (str): Path to the second PDB file.
    chain_type_list (list): List of chain identifiers to analyze.
    calculate_region (str): Region to calculate RMSD ("CDR", "FR", "ALL").

    Returns:
    dict: RMSD results for specified regions.
    """
    def chain_name_replace(name):
        return 'H' if name == 'A' else 'L'
    
    
    # Extract sequence and coordinate data for both PDB files
    chain_sequences_1, chain_coords_1 = extract_chain_sequence_coord(
                                                                file_1, 
                                                                chain_id_list=raw_chain_type_list, 
                                                                coord_type=coord_type
                                                            )
    chain_sequences_2, chain_coords_2 = extract_chain_sequence_coord(
                                                                file_2, 
                                                                chain_id_list=gen_chain_type_list, 
                                                                coord_type=coord_type
                                                            )

    rmsd_results = {}  # Save results as dict.
    print(chain_sequences_2)
    raw_start_idx = 0
    for raw_chain_type, gen_chain_type in zip(raw_chain_type_list, gen_chain_type_list):
        # Extract sequences and coordinates for the current chain
        seq_1 = chain_sequences_1[raw_chain_type]
        coord_dict_1 = chain_coords_1[raw_chain_type]
        
        # We may meet some mask ratio not equal to 1.
        # Need to visit the json file extract the seq_2;
        # And then replace the seq_2 for convient get
        # the CDR idx.
        if masked_ratio != 1.0:
            seq_2 = backup_seq_dict[gen_chain_type]
        else:
            seq_2 = chain_sequences_2[gen_chain_type]
        coord_dict_2 = chain_coords_2[gen_chain_type]
        # if predict_type == 'light' or predict_type == 'scfv':
        raw_start_idx = find_start_idx(seq_1, seq_2)
        if raw_start_idx is not None and verbose:
            print(f"The starting index is {raw_start_idx}.")
        else:
            print("No match found.")
            print(f'seq1: {seq_1}')
            print(f'seq2: {seq_2}')
        seq_1 = seq_1[raw_start_idx:raw_start_idx+len(seq_2)]
        # We need to make sure that exclute the 
        # X residues, other residues is equal between
        # of the seq1 and seq2.
        equal_state, equal_result = compare_seq_equal(seq_1, seq_2)
        if not equal_state:
            print(equal_result)
            continue        


        # Get CDR lists and variable sequence length
        cdr1_list, cdr2_list, cdr3_list = find_x_intervals_as_ranges(seq_2)
        var_seq1 = seq_1
        var_seq2 = seq_2
        print(var_seq1)
        print(var_seq2)
        min_var_seq = min(len(var_seq1), len(var_seq2))
        total_indices = set(range(1, min_var_seq + 1))  # Use var_seq to determine length
        cdr_indices = set(list(cdr1_list) + list(cdr2_list) + list(cdr3_list))
        fr_list = sorted(total_indices - cdr_indices)

        # Flatten coordinates for each region
        # list(chain.from_iterable(coord_dict1[idx] for idx in index_list))
        def get_coords(coord_dict1, coord_dict2, index_list, raw_idx):
            # Need notice that coord_dict1 must be the raw coord.
            coord1_select, coord2_select = [], []
            for idx in index_list:
                min_len = min(len(coord_dict1[raw_idx+idx]), len(coord_dict2[idx]))
                coord1_select.extend(coord_dict1[raw_idx+idx][:min_len])
                coord2_select.extend(coord_dict2[idx][:min_len])
            return coord1_select, coord2_select

        cdr1_coords_1, cdr1_coords_2 = get_coords(coord_dict_1, coord_dict_2, cdr1_list, raw_start_idx) 
        cdr2_coords_1, cdr2_coords_2 = get_coords(coord_dict_1, coord_dict_2, cdr2_list, raw_start_idx)
        cdr3_coords_1, cdr3_coords_2 = get_coords(coord_dict_1, coord_dict_2, cdr3_list, raw_start_idx)
        fr_coords_1, fr_coords_2 = get_coords(coord_dict_1, coord_dict_2, fr_list, raw_start_idx)
        all_coords_1 = cdr1_coords_1 + cdr2_coords_1 + cdr3_coords_1 + fr_coords_1
        all_coords_2 = cdr1_coords_2 + cdr2_coords_2 + cdr3_coords_2 + fr_coords_2

        # Calculate RMSD for specified region
        if calculate_region == "CDR":
            if rmsd_type == 'af3_local':
                rmsd_cdr1 = af3_rmsd_local(np.array(cdr1_coords_1), np.array(cdr1_coords_2))
                rmsd_cdr2 = af3_rmsd_local(np.array(cdr2_coords_1), np.array(cdr2_coords_2))
                rmsd_cdr3 = af3_rmsd_local(np.array(cdr3_coords_1), np.array(cdr3_coords_2))
            elif rmsd_type == 'af3_global':
                rmsd_cdr1 = af3_rmsd_global(
                                np.array(all_coords_1),
                                np.array(all_coords_2),
                                np.array(cdr1_coords_1), 
                                np.array(cdr1_coords_2)
                            )
                rmsd_cdr2 = af3_rmsd_global(
                                np.array(all_coords_1),
                                np.array(all_coords_2),
                                np.array(cdr2_coords_1), 
                                np.array(cdr2_coords_2)
                            )
                rmsd_cdr3 = af3_rmsd_global(
                                np.array(all_coords_1),
                                np.array(all_coords_2),
                                np.array(cdr3_coords_1), 
                                np.array(cdr3_coords_2),
                            )
            else:
                raise AttributeError(f'The calculation type of RMSD: {rmsd_type} is not supported.')
            rmsd_results[f"{gen_chain_type}_CDR1"] = rmsd_cdr1      # Here using the gen_chain_type is convient for heavy or light.
            rmsd_results[f"{gen_chain_type}_CDR2"] = rmsd_cdr2
            rmsd_results[f"{gen_chain_type}_CDR3"] = rmsd_cdr3
        
        # Below functions not valid. HAVE BUG.
        
        elif calculate_region == "FR":
            rmsd_results[f"{gen_chain_type}_FR"] = af3_rmsd_local(np.array(fr_coords_1), np.array(fr_coords_2))
        elif calculate_region == "ALL":
            rmsd_results[f"{gen_chain_type}_ALL"] = af3_rmsd_local(np.array(all_coords_1), np.array(all_coords_2))
        else:
            raise ValueError(f"Invalid region specified: {calculate_region}")

    return rmsd_results


def save_rmsd_to_csv(results, output_csv_path):
    """
    Save RMSD results to a CSV file.

    Parameters:
    results (list of dict): Each dict contains 'subdir' and RMSD values.
    output_csv_path (str): Path to the output CSV file.
    """
    # Define column headers for CSV
    columns = ["subdir", "H1", "H2", "H3", "L1", "L2", "L3"]

    # Create a DataFrame
    df = pd.DataFrame(results, columns=columns)

    # Save DataFrame to CSV
    df.to_csv(output_csv_path, index=False)


def append_column_means_to_csv(csv_path):
    """
    Calculate and append column means to a CSV file.

    Parameters:
    csv_path (str): Path to the CSV file.
    """
    # Load the CSV into a DataFrame
    df = pd.read_csv(csv_path)

    # Calculate means for RMSD columns (exclude "subdir" and "Mean")
    mean_row = df.loc[:, "A_CDR1":"B_CDR3"].mean()

    # Create a new row for column means
    mean_row = pd.DataFrame([mean_row], columns=df.columns[1:-1])
    mean_row.insert(0, "subdir", "Mean")
    mean_row["Mean"] = mean_row.mean(axis=1)

    # Append the mean row to the DataFrame
    df = pd.concat([df, mean_row], ignore_index=True)

    # Save the updated DataFrame back to the CSV
    df.to_csv(csv_path, index=False)
    

def create_csv_fpath(base_dir, name='local_rmsd.csv'):
    parent_dir = os.path.dirname(base_dir)
    return os.path.join(parent_dir, name)


def compare_seq_equal(seq1, seq2):
    """
    Remove positions with 'X' in seq2 from both seq1 and seq2,
    and check if the resulting sequences are identical.
    
    Args:
        seq1 (str): First sequence.
        seq2 (str): Second sequence (contains 'X').
    
    Returns:
        bool: True if the processed sequences are identical, False otherwise.
        str: The processed sequences if not identical.
    """
    if len(seq1) != len(seq2):
        raise ValueError(f"Sequences must have the same length:\nseq1{seq1}\nseq2{seq2}")
    
    # Filter out positions with 'X' in seq2
    seq1_filtered = ''.join(a for a, b in zip(seq1, seq2) if b != 'X')
    seq2_filtered = ''.join(b for b in seq2 if b != 'X')
    
    # Compare filtered sequences
    if seq1_filtered == seq2_filtered:
        return True, ""
    else:
        return False, f"Processed seq1: {seq1_filtered}\nProcessed seq2: {seq2_filtered}"


def find_start_idx(seq1, seq2):
    """
    Extract the first segment from seq2 before 'X' and find its starting index in seq1.

    Args:
        seq1 (str): The sequence where alignment is searched.
        seq2 (str): The reference sequence with 'X' placeholders.

    Returns:
        int: The 1-based index of the starting position of the first non-'X' segment in seq1.
             If no match is found, returns None.
    """
    # Extract the first segment in seq2 before 'X'
    prefix = seq2.split('X', 1)[0]
    
    # Find the starting index of the prefix in seq1
    start_idx = seq1.find(prefix)
    if start_idx != -1:
        return start_idx 
    return None


def determine_protein_type(filename):
    """
    Determines the protein type based on specific rules for two independent letters in the filename.

    Args:
        filename (str): The filename to analyze.

    Returns:
        str: The protein type ('ab', 'nb', 'light', 'scfv', or 'unknown').
    """
    # 通过 "_" 分割文件名
    parts = filename.split('_')
    
    # 确保分割后长度足够
    if len(parts) < 3:
        return "unknown"

    # 提取两个字母
    first_letter = parts[1]  # idx 为 1 的位置
    second_letter = parts[2]  # idx 为 2 的位置

    # 判断规则
    if first_letter.isupper() and second_letter.isupper():
        return "ab"  # 两个字母都存在且大写
    elif first_letter.isupper() and second_letter == "":
        return "nb"  # 只有第一个字母且大写
    elif second_letter.isupper() and first_letter == "":
        return "light"  # 只有第二个字母且大写
    elif first_letter.isupper() and second_letter.islower():
        return "scfv"  # 两个字母，第一个大写，第二个小写
    else:
        return "unknown"  # 无法匹配

def construct_back_seq_dict(back_dict, raw_chain_list):
    # extraction H and L mask.
    backup_dict = {}
    for chain in raw_chain_list:
        
        backup_dict[chain] = raw_chain_list
    

def main():
    save_csv = False
    masked_ratio = 0.3
    align_region = 'global'    # choices, ['global', 'local']
    
    with open("/mnt/nas-new/home/yangnianzu/icml/data/all_structures/processed/" \
              "comprehensive_processed_data_with_protein_antigen_and_without_antigen.json", "r") as f:
        backup_allseqs_dict = json.load(f) 
    
    # Need to notice that the raw file must using the chothia, 
    # beacuse it has split a new name for the scfv files.
    # Like ['E', 'e']
    raw_file_dir = '/mnt/nas-new/home/yangnianzu/icml/data/all_structures/chothia/'
    # multi_chain_dir = '/mnt/nas-new/home/yangnianzu/icml/results/boltz/boltz_results_normal_ab_without_antigen/predictions'
    
    boltz_results_dirpath = '/mnt/nas-new/home/yangnianzu/icml/results/boltz/masked_probability_0.3'
    sub_results_dirs = get_subdirectories(boltz_results_dirpath)
    # sub_results_dirs = ['/mnt/nas-new/home/yangnianzu/icml/results/boltz/boltz_results_nanobody/predictions']
    for sub_dir in sub_results_dirs:
        sub_dir_path = os.path.join(boltz_results_dirpath, sub_dir+'/predictions')
        # sub_dir_path = sub_results_dirs[0]
        
        if align_region == 'local':
            output_csv = create_csv_fpath(sub_dir_path)
            rmsd_type = 'af3_local'
        elif align_region == 'global':
            output_csv = create_csv_fpath(sub_dir_path, name='global_rmsd.csv')
            rmsd_type = 'af3_global'
        else:
            raise AttributeError(f'The align region only contains \'local\' and \'global\' .')
        
        rmsd_results = []
        # Now we first consider about the antibody rmsd.
        for sub_dir in get_subdirectories(sub_dir_path):

            sub_dir_split_list = sub_dir.split('_')
            
            # For different mask ratio.
            backup_key = '_'.join(sub_dir.split('_')[:4])
            backup_dict = backup_allseqs_dict[backup_key]
            
            predict_type = determine_protein_type(sub_dir)
            pdb_name = sub_dir_split_list[0]
            raw_pdb_fpath = os.path.join(raw_file_dir, pdb_name+'.pdb')
            
            target_directory = os.path.join(sub_dir_path, sub_dir)
            gen_pdb_files = glob.glob(os.path.join(target_directory, "*.pdb"))
            # Need to cal each ten results with 
            for i, gen_pdb_fpath in enumerate(gen_pdb_files):
                gen_pdb_name = os.path.basename(gen_pdb_fpath)
                print(gen_pdb_fpath)
                
                back_seq_dict = {}
                if predict_type == 'ab':
                    antibody_chain_0, antibody_chain_1 = sub_dir_split_list[1], sub_dir_split_list[2]
                    raw_chain_list = [antibody_chain_0, antibody_chain_1]
                    gen_chain_list = ['A', 'B']
                    back_seq_dict['A'] = backup_dict['H_chain_masked_seq']
                    back_seq_dict['B'] = backup_dict['L_chain_masked_seq']
                elif predict_type == 'nb':
                    nanobody_chian = sub_dir_split_list[1]
                    raw_chain_list = [nanobody_chian]
                    gen_chain_list = ['A']
                    back_seq_dict['A'] = backup_dict['H_chain_masked_seq']
                elif predict_type == 'light':
                    light_chain = sub_dir_split_list[2]
                    raw_chain_list = [light_chain]
                    gen_chain_list = ['A']
                    back_seq_dict['A'] = backup_dict['L_chain_masked_seq']
                elif predict_type == 'scfv':
                    heavy_chain, light_chain = sub_dir_split_list[1], sub_dir_split_list[2]
                    raw_chain_list = [heavy_chain, light_chain]
                    gen_chain_list = ['A', 'B']
                    back_seq_dict['A'] = backup_dict['H_chain_masked_seq']
                    back_seq_dict['B'] = backup_dict['L_chain_masked_seq']
                # try:
                # # 计算 CDR 区域 RMSD
                # try:
                rmsd_cdr = calculate_rmsd(
                    raw_pdb_fpath,  # Need make sure that this is raw file, because need get CDR region.
                    gen_pdb_fpath,
                    raw_chain_type_list=raw_chain_list, 
                    gen_chain_type_list=gen_chain_list, # Here fix the gen chain type is A(heavy) and B(light).
                    calculate_region="CDR",
                    masked_ratio=masked_ratio,
                    backup_seq_dict=back_seq_dict,
                    rmsd_type=rmsd_type
                )
                
                # Append result as a row to the results list
                if predict_type == 'ab' or predict_type == 'scfv':
                    rmsd_results.append({
                        "subdir": gen_pdb_name,
                        "H1": rmsd_cdr.get("A_CDR1", "N/A"),
                        "H2": rmsd_cdr.get("A_CDR2", "N/A"),
                        "H3": rmsd_cdr.get("A_CDR3", "N/A"),
                        "L1": rmsd_cdr.get("B_CDR1", "N/A"),
                        "L2": rmsd_cdr.get("B_CDR2", "N/A"),
                        "L3": rmsd_cdr.get("B_CDR3", "N/A"),
                    })
                elif predict_type == 'nb':
                    rmsd_results.append({
                        "subdir": gen_pdb_name,
                        "H1": rmsd_cdr.get("A_CDR1", "N/A"),
                        "H2": rmsd_cdr.get("A_CDR2", "N/A"),
                        "H3": rmsd_cdr.get("A_CDR3", "N/A"),
                        # "L1": rmsd_cdr.get("L_CDR1", "N/A"),
                        # "L2": rmsd_cdr.get("L_CDR2", "N/A"),
                        # "L3": rmsd_cdr.get("L_CDR3", "N/A"),
                    })
                elif predict_type == 'light':
                    rmsd_results.append({
                        "subdir": gen_pdb_name,
                        # "H1": rmsd_cdr.get("H_CDR1", "N/A"),
                        # "H2": rmsd_cdr.get("H_CDR2", "N/A"),
                        # "H3": rmsd_cdr.get("H_CDR3", "N/A"),
                        "L1": rmsd_cdr.get("A_CDR1", "N/A"),
                        "L2": rmsd_cdr.get("A_CDR2", "N/A"),
                        "L3": rmsd_cdr.get("A_CDR3", "N/A"),
                    })
            
                
                print(f"{gen_pdb_name} RMSD for CDR regions:\n", rmsd_cdr)
                # except:
                #     print(f"{gen_pdb_name} has problem")
        
        # Save results to CSV
        if save_csv:
            save_rmsd_to_csv(rmsd_results, output_csv)
            print(f"RMSD results saved to {output_csv}")
# Example usage
if __name__ == "__main__":
    
    main()
    # pred_h1_fpath = '/mnt/nas-new/home/yangnianzu/icml/results/boltz/boltz_results_scfv_ab/predictions/7pa6_N_n_D_all_cdr_masked/7pa6_N_n_D_all_cdr_masked_model_0.pdb'
    # ture_h1_fpath = '/mnt/nas-new/home/yangnianzu/icml/data/all_structures/chothia/7pa6.pdb'
                
    # pred_seqs, pred_coord = extract_chain_sequence_coord(pred_h1_fpath, chain_id_list=['A', 'B'])
    # true_seqs, true_coord = extract_chain_sequence_coord(ture_h1_fpath, chain_id_list=['N', 'n'])
    # pred_coord_list = list(chain.from_iterable(v for _, v in pred_coord['A'].items()))
    # true_coord_list = list(chain.from_iterable(v for _, v in true_coord['D'].items()))
    # print(pred_coord_list)
    # print(true_coord_list)
    # rmsd = af3_rmsd(np.array(pred_coord_list), np.array(true_coord_list))
    # print(rmsd)
    # with open("/mnt/nas-new/home/yangnianzu/icml/data/all_structures/processed/comprehensive_processed_data_with_protein_antigen_and_without_antigen.json", "r") as f:
    #     data = json.load(f)
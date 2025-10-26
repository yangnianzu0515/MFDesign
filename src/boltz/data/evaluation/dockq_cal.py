# Calculate the DockQ value.

# 1. Need read the sub_pdb_files of dir.
# 2. Need get the raw corresponding chothia pdb.
# 3. extract chains as a small new pdb avoid error.
# 4. Need to construct the backbone of model 
# to compare different.
import os
import subprocess
import glob
import re
import pandas as pd
import time
import numpy as np

from Bio.PDB import PDBParser, PDBIO, Select
from Bio import PDB
from Bio.SeqUtils import seq1

from utils import get_subdirectories
from align import extract_chain_sequence_coord, find_start_idx
from EBM_dockq import Meter_Unbound_Bound


class BackboneSelect(Select):
    def __init__(self, chains_to_select=[]):
        self.chains_to_select = chains_to_select


    def accept_atom(self, atom):
        # Select atoms if they belong to the backbone and the chain is in the selected list
        if atom.get_parent().get_parent().get_id() in self.chains_to_select:
            if atom.get_name() in ['N', 'CA', 'C', 'O']:  # Backbone atoms
                return 1
            else:
                return 0
        else:
            return 1

def extract_backbone(input_pdb, output_pdb, chains_to_select):
    # Parse the input PDB file
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("structure", input_pdb)

    # Initialize PDBIO for writing the filtered structure
    io = PDBIO()
    io.set_structure(structure)

    # Write only the backbone atoms from the selected chains to the output file
    io.save(output_pdb, BackboneSelect(chains_to_select))
    

def get_merge_state(split_list):
    """judege whether need to merge according to the name lenght.

    Args:
        split_list (list): a list contain the differenct chain names

    Returns:
        bool: the antibody merge state and the antigen merge state. 
    """
    ligand_merge_state = False
    receptor_merge_state = False
    if len(split_list[1] + split_list[2]) == 2:
        ligand_merge_state = True
    
    if len(split_list[3]) > 1:
        receptor_merge_state = True
    return ligand_merge_state, receptor_merge_state


def generate_letters_from_A(length):
    """
    According to the given length to generate a list start from 'A' 
    to the specific length range.
    Args:
        length (int): the lenght of the letter。

    Returns:
        list: contain and statrt from 'A'。
    """
    if length < 1:
        return []  # 如果长度小于 1，返回空列表
    return [chr(ord('A') + i) for i in range(length)]  

def get_dockq_value(result):
    # Extract DockQ value using regex
    match = re.search(r'DockQ\s+([\d\.]+)', result.stdout)
    if match:
        dockq_value = float(match.group(1))  # Convert to float
        return dockq_value
    else:
        raise ValueError("DockQ value not found in the output")


def save_and_rename_chains(input_pdb, output_pdb, chains_to_extract, chain_id_mapping):
    """
    Extract specific chains from a PDB file, rename them using temporary IDs to avoid conflicts, 
    reorder them by the given sequence, and save to a new PDB file.

    Parameters:
    - input_pdb: Path to the input PDB file.
    - output_pdb: Path to the output PDB file.
    - chains_to_extract: List of chain IDs to extract in desired order.
    - chain_id_mapping: Dictionary mapping old chain IDs to new chain IDs.
                        e.g., {"H": "A", "L": "B"}
    """
    import os
    from Bio import PDB

    if os.path.exists(output_pdb):
        print(f"File '{output_pdb}' already exists. Skipping extraction.")
        return

    # Initialize PDB parser and I/O
    parser = PDB.PDBParser(QUIET=True)
    io = PDB.PDBIO()
    structure = parser.get_structure("protein", input_pdb)

    # Step 1: Extract chains and build a new structure
    new_structure = PDB.Structure.Structure("new_protein")
    model = PDB.Model.Model(0)  # Create a new model
    chain_map = {chain.id: chain for model in structure for chain in model}

    for chain_id in chains_to_extract:
        if chain_id in chain_map:
            chain = chain_map[chain_id]
            model.add(chain.copy())  # Use a copy to avoid modifying the original structure

    new_structure.add(model)

    # Step 2: Apply temporary renaming to avoid conflicts
    temp_mapping = {old: f"TEMP_{old}" for old in chain_id_mapping.keys()}
    for chain in model:
        if chain.id in temp_mapping:
            chain.id = temp_mapping[chain.id]  # Assign temporary IDs

    # Step 3: Apply final renaming from temporary IDs to desired IDs
    final_mapping = {f"TEMP_{old}": new for old, new in chain_id_mapping.items()}
    for chain in model:
        if chain.id in final_mapping:
            chain.id = final_mapping[chain.id]  # Assign final IDs

    # Step 4: Save the new structure
    io.set_structure(new_structure)
    io.save(output_pdb)

    print(f"Chains {', '.join(chains_to_extract)} extracted, renamed, and reordered. Saved to '{output_pdb}'.")

  
def reorder_pdb_residues(input_path, output_path):
    # Read the PDB file
    with open(input_path, 'r') as file:
        lines = file.readlines()

    # Group atoms by chain
    chains = {}
    for line in lines:
        if line.startswith('ATOM') or line.startswith('HETATM'):
            chain_id = line[21]  # Get chain ID (from column 22)
            if chain_id not in chains:
                chains[chain_id] = []
            chains[chain_id].append(line)

    # Re-number residues within each chain
    new_lines = []
    for chain_id, chain_atoms in chains.items():
        # Get residue numbers for the current chain
        residue_map = {}
        for line in chain_atoms:
            residue_num = line[22:27].strip()
            if residue_num not in residue_map:
                residue_map[residue_num] = []
            residue_map[residue_num].append(line)

        # Re-number residues for each chain
        new_residue_counter = 1  # Start residue numbering from 1 for each chain
        for residue_num in residue_map.keys():  # Sort residues by number
            for line in residue_map[residue_num]:
                # Update residue number (columns 23-26)
                # Extra space of the residue counter is for alpha.
                updated_line = line[:22] + f"{new_residue_counter:4d} " + line[27:]
                new_lines.append(updated_line)
            
            # Increment residue counter
            new_residue_counter += 1

    # Write the updated PDB file to the specified output path
    with open(output_path, 'w') as file:
        file.writelines(new_lines)

class GroundTruthExtractor(Select):
    """
    Custom Select class to save only the matched ground-truth residues.
    """
    def __init__(self, residue_ids_by_chain):
        super().__init__()
        self.residue_ids_by_chain = residue_ids_by_chain

    def accept_residue(self, residue):
        chain_id = residue.parent.id  # Get the chain ID
        return chain_id in self.residue_ids_by_chain and residue.get_id() in self.residue_ids_by_chain[chain_id]
    

def extract_ground_truth(predict_pdb, original_pdb, output_pdb, chain_id_map):
    """
    Extract ground truth residues for multiple chains based on a chain ID mapping.

    :param predict_pdb: Path to the predicted PDB file
    :param original_pdb: Path to the original PDB file
    :param output_pdb: Path to the output PDB file to save the extracted structure
    :param chain_id_map: Dictionary mapping predicted chain IDs to original chain IDs
    """
    # Parse the PDB files
    parser = PDBParser(QUIET=True)
    predict_structure = parser.get_structure("predict", predict_pdb)
    original_structure = parser.get_structure("original", original_pdb)

    residue_ids_by_chain = {}

    for predict_chain_id in chain_id_map:
        print(predict_chain_id)
        original_chain_id = predict_chain_id
        # Get predicted sequence
        predict_chain = predict_structure[0][predict_chain_id]  # Assuming model 0
        predict_sequence = "".join(
            seq1(residue.get_resname()) for residue in predict_chain if residue.id[0] == " "
        )

        # Get original sequence
        original_chain = original_structure[0][original_chain_id]
        original_sequence = "".join(
            seq1(residue.get_resname()) for residue in original_chain if residue.id[0] == " "
        )

        # Match the sequence
        # print(original_sequence)
        # print(predict_sequence)
        prefix = predict_sequence.split('X', 1)[0]
        start_index = original_sequence.find(prefix)
        if start_index == -1:
            raise ValueError(
                f"Predicted sequence for chain {predict_chain_id} not found in original chain {original_chain_id}."
            )

        # Extract matching residues
        residue_ids_by_chain[original_chain_id] = [
            residue.id for i, residue in enumerate(original_chain)
            if i in range(start_index, start_index + len(predict_sequence))
        ]

    # Save the matched residues as a new PDB file
    io = PDBIO()
    io.set_structure(original_structure)
    io.save(output_pdb, GroundTruthExtractor(residue_ids_by_chain))
    print(f"Ground truth structure extracted and saved to {output_pdb}")
          

class ReplaceUNKForChains:
    """
    Replace UNK residues in specified chains of the predict structure
    with corresponding residues from the ground truth structure.
    """
    def __init__(self, ground_truth_structure, target_chains):
        """
        :param ground_truth_structure: Parsed ground truth structure
        :param target_chains: List of chain IDs to process
        """
        self.ground_truth_structure = ground_truth_structure
        self.target_chains = set(target_chains)

    def replace_residues(self, predict_structure):
        for chain in predict_structure.get_chains():
            chain_id = chain.id
            if chain_id not in self.target_chains:  # Skip if not in target chains
                continue

            # Get the corresponding chain in ground truth
            if chain_id not in self.ground_truth_structure[0]:
                raise ValueError(f"Chain {chain_id} not found in ground truth structure.")

            ground_truth_chain = self.ground_truth_structure[0][chain_id]

            for residue in chain.get_residues():
                if residue.get_resname() == "UNK":  # Check if it's UNK
                    # Find the corresponding residue in the ground truth
                    res_id = residue.id
                    if res_id in ground_truth_chain:
                        ground_truth_residue = ground_truth_chain[res_id]
                        # Replace the residue name
                        residue.resname = ground_truth_residue.get_resname()
                        

def replace_unk_in_chains(predict_pdb, ground_truth_pdb, output_pdb, target_chains):
    """
    Replace all UNK residues in specified chains of the predict PDB file
    with the corresponding residues from the ground truth PDB file.

    :param predict_pdb: Path to the predicted PDB file
    :param ground_truth_pdb: Path to the ground truth PDB file
    :param output_pdb: Path to save the modified PDB file
    :param target_chains: List of chain IDs to process
    """
    parser = PDBParser(QUIET=True)
    predict_structure = parser.get_structure("predict", predict_pdb)
    ground_truth_structure = parser.get_structure("ground_truth", ground_truth_pdb)

    # Replace UNK residues for target chains
    replacer = ReplaceUNKForChains(ground_truth_structure, target_chains)
    replacer.replace_residues(predict_structure)

    # Save the updated structure
    io = PDBIO()
    io.set_structure(predict_structure)
    io.save(output_pdb)
    print(f"Updated PDB file saved to {output_pdb}")

def get_coords(coord_dict1, coord_dict2, index_list, raw_idx):
    # Need notice that coord_dict1 must be the raw coord.
    coord1_select, coord2_select = [], []
    for idx in index_list:
        min_len = min(len(coord_dict1[raw_idx+idx]), len(coord_dict2[idx]))
        coord1_select.extend(coord_dict1[raw_idx+idx][:min_len])
        coord2_select.extend(coord_dict2[idx][:min_len])
    return coord1_select, coord2_select


def main():
    only_CA = False
    deal_mask = True
    dockq_bin_path = '/mnt/nas-new/home/yangnianzu/icml/eval_scripts/DockQ-1.0/DockQ.py'
    native_dir = '/mnt/nas-new/home/yangnianzu/icml/data/all_structures/chothia/'
    model_dir = '/mnt/nas-new/home/yangnianzu/icml/results/boltz/' \
                'masked_probability_1.0/boltz_results_after_date_with_antigen/predictions'
    native_unmask_dir = '/mnt/nas-new/home/yangnianzu/icml/results/boltz/' \
                'masked_probability_1.0_GT_seq/boltz_results_after_date_with_antigen/predictions'
    perm_new_save_dir = '/mnt/nas-new/home/yangnianzu/icml/eval_scripts/sub_pred_unmask_chains_pdb/'
                
    # May we need a new sub chains from the Polymer
        
    # TODO: Carefully, need to specific the save json name according to different situations.
    save_dockq_results_fpath = '/mnt/nas-new/home/yangnianzu/icml/results/boltz/verify/dockq_comapre_unmask_mask_backbone.csv'           
    # command = ['python', dockq_bin_path,]        
    
    columns = ["subdir", "dockq1"]    
    fail_pdb_name_set = set()
    # Create a DataFrame
    # df = pd.DataFrame(results, columns=columns)
    results = []
    for sub_dir in get_subdirectories(model_dir):
        # print(sub_dir)
        target_directory = os.path.join(model_dir, sub_dir)
        gen_pdb_files = glob.glob(os.path.join(target_directory, "*.pdb"))
        for i, gen_pdb_fpath in enumerate(gen_pdb_files):
            pred_pdb_name = os.path.basename(gen_pdb_fpath)
            

            split_list = pred_pdb_name.split('_')
            # print(split_list)
            merge_ligand, merge_receptor = get_merge_state(split_list)
            print(merge_ligand, merge_receptor)

            all_chain_len = len(split_list[1]+split_list[2]+split_list[3])
            gen_chain_list = generate_letters_from_A(all_chain_len)

            pdb_name = split_list[0]
            native_unmask_dir_target = os.path.join(native_unmask_dir, '_'.join(sub_dir.split('_')[:4]))
            new_umask_pdb_name = '_'.join(split_list[:4]) + '_model_' + f'{split_list[8]}'
            print(new_umask_pdb_name)
            raw_pdb_fpath = os.path.join(native_unmask_dir_target, new_umask_pdb_name)
            # raw_pdb_fpath = os.path.join(native_dir, pdb_name+'.pdb')
            # Merge the antibody
            if merge_ligand:
                native_lig_chains_list = [split_list[1], split_list[2]]
                model_lig_chains_list = gen_chain_list[:2]

            else:
                native_lig_chains_list = [split_list[1] if split_list[1] != '' else split_list[2]]
                model_lig_chains_list = gen_chain_list[:1]
            # Merge the antigens
            if merge_receptor:
                native_rec_chains_list = list(split_list[3])
                model_rec_chains_list = [name for name in gen_chain_list 
                                        if name not in model_lig_chains_list]
                
            else:
                native_rec_chains_list = list(split_list[3])
                model_rec_chains_list = [name for name in gen_chain_list 
                                        if name not in model_lig_chains_list]
                
            # Need to construct id mapping.
            id_mapping = {}
            native_whole_list = native_lig_chains_list + native_rec_chains_list
            model_whole_list = model_lig_chains_list + model_rec_chains_list
            for k, v in zip(native_whole_list, model_whole_list):
                # print(k, v)
                id_mapping[k] = v
            
            # Create chain_map dict.
            sub_raw_pdb_fpath = raw_pdb_fpath
            # sub_raw_pdb_fpath = os.path.join(perm_new_save_dir, pdb_name+'.pdb')
            # # extract sub chain as new for perm.
            # save_and_rename_chains(
            #     raw_pdb_fpath, 
            #     sub_raw_pdb_fpath, 
            #     chains_to_extract=native_lig_chains_list+native_rec_chains_list,
            #     chain_id_mapping=id_mapping
            # )
            
            execute_command = [dockq_bin_path, gen_pdb_fpath, sub_raw_pdb_fpath, '-perm1']
                
            if only_CA:
                execute_command.extend(['-useCA'])

            model_chain1 = ['-model_chain1']
            model_chain1.extend(model_lig_chains_list)
            native_chain1 = ['-native_chain1']
            native_chain1.extend(model_lig_chains_list)
            
            model_chain2 = ['-model_chain2']
            model_chain2.extend(model_rec_chains_list)
            native_chain2 = ['-native_chain2']
            native_chain2.extend(model_rec_chains_list)
            execute_command = execute_command + model_chain1 + native_chain1 + model_chain2 + native_chain2
             
            sub_reorder_pdb_fpath = os.path.join(perm_new_save_dir, pdb_name+'_reorder.pdb')
            if not os.path.exists(sub_reorder_pdb_fpath):
                reorder_pdb_residues(sub_raw_pdb_fpath, sub_reorder_pdb_fpath)
            
            # Now we need to remove the side atoms.
            # Because the DockQ calculate the fnat between all atoms.
            # and UNK residues only have five atoms [N, CA, C, O, CB].
            # we only consider about backbone.
            # The side atoms of antigens are maintained.    
            # Firse is native
            sub_reorder_remove_side_pdb_fpath = os.path.join(perm_new_save_dir, new_umask_pdb_name+'_anti_backbone.pdb')
            extract_backbone(sub_reorder_pdb_fpath, sub_reorder_remove_side_pdb_fpath, chains_to_select=model_lig_chains_list)
            
            # Second is model.
            gen_remove_side_pdb_fpath = os.path.join(os.path.dirname(gen_pdb_fpath), f'anti_backbone_{pred_pdb_name}')
            extract_backbone(gen_pdb_fpath, gen_remove_side_pdb_fpath, chains_to_select=model_lig_chains_list)
            
            # Third Need to extract ground truth.
            # extract_ground_truth(
            #     gen_remove_side_pdb_fpath,
            #     sub_reorder_remove_side_pdb_fpath,
            #     sub_reorder_remove_side_pdb_fpath,
            #     model_lig_chains_list+model_rec_chains_list,
            # )
            
            if not deal_mask:
                execute_command[1] = gen_remove_side_pdb_fpath
                
            else:
                # need to rename the UNK residues.
                gen_rename_side_pdb_fpath = os.path.join(os.path.dirname(gen_pdb_fpath), f'rename_{pred_pdb_name}')
                replace_unk_in_chains(
                    gen_remove_side_pdb_fpath,
                    sub_reorder_remove_side_pdb_fpath,
                    gen_rename_side_pdb_fpath,
                    target_chains=model_lig_chains_list
                )
                execute_command[1] = gen_rename_side_pdb_fpath
                
            execute_command[2] = sub_reorder_remove_side_pdb_fpath   
            
            try:
                print(execute_command)
                dockq_result = subprocess.run(
                    execute_command, 
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                    
                dockq_value = get_dockq_value(dockq_result)
                print(f'{pred_pdb_name} DockQ value: {dockq_value}')
                dockq_results_dict = {
                    'subdir': pred_pdb_name,
                    'dockq1': dockq_value,
                }
                os.remove(gen_remove_side_pdb_fpath)
                os.remove(gen_rename_side_pdb_fpath)
            except subprocess.CalledProcessError as e:
                print(f"Command failed with return code {e.returncode}")
                # print(f"Standard output:\n{e.stdout}")
                print(f"Error output:\n{e.stderr}")

                fail_pdb_name_set.add(pdb_name)
                os.remove(gen_remove_side_pdb_fpath)
                os.remove(gen_rename_side_pdb_fpath)
            #     continue
        
            results.append(dockq_results_dict)
            print('---------------------------------')
            time.sleep(1.2)
            
    df = pd.DataFrame(results, columns=columns)
    df.to_csv(save_dockq_results_fpath)
    print(f'Finally failded pdb name: {fail_pdb_name_set}')
    

def ebm_main():
    
    save_csv_results_dir = '/mnt/nas-new/home/yangnianzu/icml/results/boltz/verify'
    
    # Paramters.
    CA = False
    backbone = True
    extract = False   # Control whether we need to extract the results.
    native_type = 'predict'  # Choice. ['predict', 'raw']
    
    # Different path for compare.
    model_dpath = '/mnt/nas-new/home/yangnianzu/icml/results/boltz/masked_probability_1.0/boltz_results_after_date_with_antigen/predictions'
    native_dpath = '/mnt/nas-new/home/yangnianzu/icml/results/boltz/masked_probability_1.0_GT_seq/boltz_results_after_date_with_antigen/predictions'
    
    if backbone:
        coord_type = 'backbone'
    elif CA:
        coord_type = 'CA'
    else:
        raise KeyError('Coord type only support backbone and CA.')
    
    # Save csv fpath.
    save_csv_fpath = os.path.join(save_csv_results_dir, 'ebm_dockq_mask_unmask.csv')
    columns = ["subdir", "dockq1"]    
    
    # Dockq cal Object.
    Dockq_meter = Meter_Unbound_Bound()
    
    results = []
    for sub_dir in get_subdirectories(model_dpath):
        
        target_dir = os.path.join(model_dpath, sub_dir)
        gen_pdb_files = glob.glob(os.path.join(target_dir, "*.pdb"))
        for i, gen_pdb_fpath in enumerate(gen_pdb_files):
            pred_pdb_name = os.path.basename(gen_pdb_fpath)
            pdb_name_split_list = pred_pdb_name.split('_')
            # print(split_list)
            merge_ligand, merge_receptor = get_merge_state(pdb_name_split_list)
            print(merge_ligand, merge_receptor)

            all_chain_len = len(
                            pdb_name_split_list[1] +
                            pdb_name_split_list[2] +
                            pdb_name_split_list[3]
                        )
            gen_chain_list = generate_letters_from_A(all_chain_len)

            pdb_name = pdb_name_split_list[0]
            
            # Here we need consider two situation,
            # First native is predicted.
            # Second native is raw.
            if native_type == 'raw':
                raw_pdb_fpath = os.path.join(native_dpath, pdb_name+'.pdb')
            elif native_type == 'predict':
                native_dir_target = os.path.join(native_dpath, '_'.join(sub_dir.split('_')[:4]))
                new_pdb_name = '_'.join(pdb_name_split_list[:4]) + '_model_' + f'{pdb_name_split_list[8]}'
                raw_pdb_fpath = os.path.join(native_dir_target, new_pdb_name)
            else:
                raise AttributeError(f'natiev_type {native_type} is wrong.')
          
            # Merge the antibody
            if merge_ligand:
                native_lig_chains_list = [pdb_name_split_list[1], pdb_name_split_list[2]]
                model_lig_chains_list = gen_chain_list[:2]

            else:
                native_lig_chains_list = [
                                        pdb_name_split_list[1] 
                                        if pdb_name_split_list[1] != '' 
                                        else pdb_name_split_list[2]
                                    ]
                model_lig_chains_list = gen_chain_list[:1]
            # Merge the antigens
            native_rec_chains_list = list(pdb_name_split_list[3])
            model_rec_chains_list = [name for name in gen_chain_list 
                                    if name not in model_lig_chains_list]
            
            # If we do not extract the chains. 
            # we only keep chain name is same.
      
            if not extract:
                native_lig_chains_list = model_lig_chains_list
                native_rec_chains_list = model_rec_chains_list

            # Need to constuct mapping.
            chain_mapping = {}
            native_whole_list = native_lig_chains_list + native_rec_chains_list
            model_whole_list = model_lig_chains_list + model_rec_chains_list
            for k, v in zip(native_whole_list, model_whole_list):
                # print(k, v)
                chain_mapping[k] = v
                
            native_complex_pdb_seq, native_complex_pdb_coord = extract_chain_sequence_coord(
                raw_pdb_fpath,
                chain_id_list=native_lig_chains_list+native_rec_chains_list,
                coord_type=coord_type
            )    
                
            model_complex_pdb_seq, model_complex_pdb_coord = extract_chain_sequence_coord(
                gen_pdb_fpath,
                chain_id_list=model_lig_chains_list+model_rec_chains_list,
                coord_type=coord_type
            )
            print(f'PDB {new_pdb_name} Dockq')
            # We need to make sure there input atoms is equal.
            lig_pred = []
            lig_true = []
            for chain in model_lig_chains_list:
                
                native_lig_seq = native_complex_pdb_seq[chain_mapping[chain]]
                model_lig_seq = model_complex_pdb_seq[chain]
                start_lig_idx = find_start_idx(native_lig_seq, model_lig_seq)
                idx_lig_list = list(range(1, len(model_lig_seq)+1))
                
                native_lig_coord = native_complex_pdb_coord[chain_mapping[chain]]
                model_lig_coord = model_complex_pdb_coord[chain]
                native_equal_lig_coord, model_equal_lig_coord = get_coords(
                    native_lig_coord,
                    model_lig_coord,
                    idx_lig_list,
                    start_lig_idx
                )
                
                lig_pred.extend(model_equal_lig_coord)
                lig_true.extend(native_equal_lig_coord)
                
            receptor_pred = []
            receptor_true = []
            for chain in model_rec_chains_list:
                
                native_recep_seq = native_complex_pdb_seq[chain_mapping[chain]]
                model_recep_seq = model_complex_pdb_seq[chain]
                
                start_recep_idx = find_start_idx(native_recep_seq, model_recep_seq)
                idx_recep_list = list(range(1, len(model_recep_seq)+1))
                
                native_recep_coord = native_complex_pdb_coord[chain_mapping[chain]]
                model_recep_coord = model_complex_pdb_coord[chain]
                native_equal_recep_coord, model_equal_recep_coord = get_coords(
                    native_recep_coord,
                    model_recep_coord,
                    idx_recep_list,
                    start_recep_idx
                )
                
                receptor_pred.extend(model_equal_recep_coord)
                receptor_true.extend(native_equal_recep_coord)
            
            # Here we need to cat as N*3.
            lig_pred_np = np.array(lig_pred).reshape(-1, 3)
            lig_true_np = np.array(lig_true).reshape(-1, 3)
            
            receptor_pred_np = np.array(receptor_pred).reshape(-1, 3)
            receptor_true_np = np.array(receptor_true).reshape(-1, 3)

            try:
                _, dockq_value = Dockq_meter.update_rmsd(
                    ligand_coors_pred=lig_pred_np,
                    receptor_coors_pred=receptor_pred_np,
                    ligand_coors_true=lig_true_np,
                    receptor_coors_true=receptor_true_np
                )
            except ValueError:
                dockq_value = 0.00
                
            single_dockq_dict = {
                    'subdir': pred_pdb_name,
                    'dockq1': dockq_value,
                }
            print(f'{dockq_value}')
            results.append(single_dockq_dict)
            
            
    df = pd.DataFrame(results, columns=columns)
    df.to_csv(save_csv_fpath)
    
            
if __name__ == '__main__':
    # main()
    ebm_main()

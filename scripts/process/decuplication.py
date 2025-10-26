import os
import numpy as np
import json
from abnumber import Chain
import pandas as pd
from itertools import islice
from tqdm import tqdm
import datetime
from joblib import Parallel, delayed
import logging
import argparse

# calculate the levenshtein distance between two sequences
def levenshtein_distance(seq1, seq2):
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = np.zeros((size_x, size_y))
    for x in range(size_x):
        matrix[x, 0] = x
    for y in range(size_y):
        matrix[0, y] = y

    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq1[x-1] == seq2[y-1]:
                matrix[x, y] = min(
                    matrix[x-1, y] + 1,
                    matrix[x-1, y-1],
                    matrix[x, y-1] + 1
                )
            else:
                matrix[x, y] = min(
                    matrix[x-1, y] + 1,
                    matrix[x-1, y-1] + 1,
                    matrix[x, y-1] + 1
                )
    return matrix[size_x - 1, size_y - 1]


# get CDR sequences according to the original sequence and the masked sequence
def get_cdr(original_seq, masked_seq):
    # find all X positions
    x_indices = [i for i, char in enumerate(masked_seq) if char == 'X']
    
    # find breakpoints (the positions where the distance between two Xs is greater than 1)
    breaks = []
    for i in range(1, len(x_indices)):
        if x_indices[i] - x_indices[i-1] > 1:
            breaks.append(i)
            
    # split x_indices into three groups according to the breakpoints
    start1 = 0
    end1 = breaks[0]
    start2 = breaks[0]
    end2 = breaks[1]
    start3 = breaks[1]
    end3 = len(x_indices)
    
    # get the corresponding CDR sequences
    cdr1 = original_seq[x_indices[start1]:x_indices[end1-1]+1]
    cdr2 = original_seq[x_indices[start2]:x_indices[end2-1]+1]
    cdr3 = original_seq[x_indices[start3]:x_indices[end3-1]+1]
    
    return cdr1, cdr2, cdr3

def decuplicate(pdb_id, group_df, distance_threshold):
    log_content = []
    sorted_group_df = group_df.sort_values(by='H_chain_seq', key=lambda x: x.str.len(), ascending=False)
    reserved_entry_list = [sorted_group_df.iloc[0]]
    assert Chain(sorted_group_df.iloc[0]['H_chain_seq'], scheme='chothia').is_heavy_chain()
    for _, row in islice(sorted_group_df.iterrows(), 1, None):  # skip the first one
        current_seq = row['H_chain_seq']
        cdr1, cdr2, cdr3 = get_cdr(current_seq, row['H_chain_masked_seq'])
        concat_seq = cdr1 + cdr2 + cdr3
        should_add = True
        for item in reserved_entry_list:
            item_cdr1, item_cdr2, item_cdr3 = get_cdr(item['H_chain_seq'], item['H_chain_masked_seq'])
            item_concat_seq = item_cdr1 + item_cdr2 + item_cdr3
            distance = levenshtein_distance(concat_seq, item_concat_seq)
            if distance <= distance_threshold:
                should_add = False
                break
        if should_add:
            reserved_entry_list.append(row)
            reference_chain = Chain(item['H_chain_seq'], scheme='chothia')
            entry_chain = Chain(row['H_chain_seq'], scheme='chothia')
            alignment = reference_chain.align(entry_chain)
            log_content.append(f'A reserved entry is found for PDB {pdb_id}:')
            log_content.append(f'Reference: {item["file_name"]}')
            log_content.append(f'Entry: {row["file_name"]}')
            log_content.append(f'Distance is greater than threshold {distance_threshold}: {distance}')
            for line in alignment.format().split('\n'):
                log_content.append(line)
            log_content.append('='*50)
    return reserved_entry_list, log_content

def main(args):
    
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(args.output_dir, 'logs-for-decuplication', f'distance_threshold_{args.distance_threshold}', current_time)
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f'decuplication_{current_time}.log')

    logging.basicConfig(filename=log_path, 
                        level=logging.WARNING, 
                        format='%(asctime)s - %(levelname)s - %(message)s', 
                        filemode='a')
   
    distance_threshold = args.distance_threshold
    
    output_file_name = f'{args.output_file_name}-distance_threshold_{distance_threshold}'

    original_data_path = args.original_data_path
    with open(original_data_path, 'r') as f:
        original_data = pd.read_csv(f)
        
    all_reserved_entry_list = list()

    grouped_data = original_data.groupby('pdb')
    results_along_with_log = Parallel(n_jobs=-1)(delayed(decuplicate)(pdb_id, group_df, distance_threshold) for pdb_id, group_df in tqdm(grouped_data, desc='In Decuplication', dynamic_ncols=True))
    all_reserved_entry_list = [item for (reserved_entry_list, log_content) in results_along_with_log for item in reserved_entry_list]
    full_log_content = [log_content for (reserved_entry_list, log_content) in results_along_with_log]

    print(f"Distance threshold: {distance_threshold}")
    logging.error(f"Distance threshold: {distance_threshold}")
    print(f"Oringnal data has {len(original_data)} entries.")
    logging.error(f"Oringnal data has {len(original_data)} entries.")
    print(f"There are {len(original_data['pdb'].unique())} PDBs.")
    logging.error(f"There are {len(original_data['pdb'].unique())} PDBs.")
    print(f"There are {len(all_reserved_entry_list)} entries reserved.")
    logging.error(f"There are {len(all_reserved_entry_list)} entries reserved.")
    print(f"We remove {len(original_data) - len(all_reserved_entry_list)} entries.")
    logging.error(f"We remove {len(original_data) - len(all_reserved_entry_list)} entries.")

    for log_content in full_log_content:
        if len(log_content) > 0:
            for line in log_content:
                logging.warning(line)
            
    all_reserved_df = pd.DataFrame(all_reserved_entry_list)
    all_reserved_df.to_csv(os.path.join(args.output_dir, f'{output_file_name}.csv'), index=False)
    print(f'decuplication is done!\nThe result is saved in {os.path.join(args.output_dir, f"{output_file_name}.csv")}.')

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--distance_threshold", type=int, default=9)
    parser.add_argument("--original_data_path", type=str, default="./data/summary.csv")
    parser.add_argument("--output_dir", type=str, default="./data")
    parser.add_argument("--output_file_name", type=str, default="summary-decuplication") # no extension
    args = parser.parse_args()
    main(args)
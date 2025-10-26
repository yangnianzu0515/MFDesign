import os
import json
from ruamel.yaml import YAML
from tqdm import tqdm
import argparse
import pandas as pd
from pathlib import Path
import shutil

"""
This script is used to batch generate yaml files for the unmasked sequences which are reversed after decuplication and saved in csv format.

In the csv file, we only reverse the antibodies with antigen. They all have heavy chain.
"""

"""
example: 
python batch_generate_yaml_for_csv_after_decuplication.py 
"""

# convert a single entry to yaml
def convert_single_entry_to_yaml(entry_value, version = 1):

    yaml_content = {'version': version, 'sequences': []}
    

    H_chain_sub_dict = {'id': entry_value['H_chain_id'], 'sequence': entry_value['H_chain_seq']}
    yaml_content['sequences'].append({'protein': H_chain_sub_dict})
        
    if not pd.isna(entry_value['L_chain_id']):
        L_chain_sub_dict = {'id': entry_value['L_chain_id'], 'sequence': entry_value['L_chain_seq']}
        yaml_content['sequences'].append({'protein': L_chain_sub_dict})
        
    antigen_seqs = eval(entry_value['antigen_seq'])
    
    
    for antigen_chain_id in eval(entry_value['antigen_chain_id']):
        antigen_sub_dict = {'id': antigen_chain_id, 'sequence': antigen_seqs[antigen_chain_id]}
        yaml_content['sequences'].append({'protein': antigen_sub_dict})

    return yaml_content


# save a single entry to yaml
def convert_to_yaml(entry_key, entry_yaml_content, save_folder='./'):
    yaml = YAML()
    yaml.indent(mapping=2, sequence=4, offset=2) # to keep consistency with the example yaml file provided by boltz
    yaml.width = 4096
    with open(os.path.join(save_folder, f"{entry_key}.yaml"), "w") as f:
        yaml.dump(entry_yaml_content, f)

def main(args):

    processed_data_dir = args.processed_data_dir_after_decuplication
    
    processed_data_dir = Path(processed_data_dir)
    
    out_dir = Path(args.output_dir)
    
    if os.path.exists(out_dir):
        print(f"The output directory {out_dir} already exists, remove it and will conduct re-generation.")
        shutil.rmtree(out_dir)
    
    out_dir.mkdir(parents=True)
    
    processed_data = pd.read_csv(processed_data_dir)

    for idx, row in tqdm(processed_data.iterrows(), total=len(processed_data), desc="Generating YAML files"):
        entry_key = row['file_name']
        entry_yaml_content = convert_single_entry_to_yaml(row)
        convert_to_yaml(entry_key, entry_yaml_content, out_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--processed_data_dir_after_decuplication', type=str, default='./data/summary-decuplication-distance_threshold_9.csv')
    parser.add_argument('--output_dir', type=str, default='./data/yaml_for_data_after_decuplication-distance_threshold_9')
    args = parser.parse_args()
    main(args)

import os
import abnumber
import pandas as pd
import json
import numpy as np
import datetime
from Bio import PDB
from Bio.PDB import Selection
from tqdm import tqdm
import logging
from joblib import Parallel, delayed
'''
This script is used to process the raw data and save the processed data to a pickle file.
'''


# ==========================================================
# Reference Codebase: DiffAb
# https://github.com/luost26/diffab
# ==========================================================


# ==========================================================
# Credit to OpenMM
# Below is the license information for OpenMM
# ==========================================================
"""
This is part of the OpenMM molecular simulation toolkit originating from
Simbios, the NIH National Center for Physics-Based Simulation of
Biological Structures at Stanford, funded under the NIH Roadmap for
Medical Research, grant U54 GM072970. See https://simtk.org.

Portions copyright (c) 2013 Stanford University and the Authors.
Authors: Peter Eastman
Contributors:

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
THE AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
# ==========================================================



# Non-standard residue substitutions
# Standard residues still use their original residue names
STANDARD_RESIDUE_SUBSTITUTIONS_INCASEOF_NON_STANDARD_RESIDUE = {
    '2AS':'ASP', '3AH':'HIS', '5HP':'GLU', 'ACL':'ARG', 'AGM':'ARG', 'AIB':'ALA', 'ALM':'ALA', 'ALO':'THR', 'ALY':'LYS', 'ARM':'ARG',
    'ASA':'ASP', 'ASB':'ASP', 'ASK':'ASP', 'ASL':'ASP', 'ASQ':'ASP', 'AYA':'ALA', 'BCS':'CYS', 'BHD':'ASP', 'BMT':'THR', 'BNN':'ALA',
    'BUC':'CYS', 'BUG':'LEU', 'C5C':'CYS', 'C6C':'CYS', 'CAS':'CYS', 'CCS':'CYS', 'CEA':'CYS', 'CGU':'GLU', 'CHG':'ALA', 'CLE':'LEU', 
    'CME':'CYS', 'CSD':'ALA', 'CSO':'CYS', 'CSP':'CYS', 'CSS':'CYS', 'CSW':'CYS', 'CSX':'CYS', 'CXM':'MET', 'CY1':'CYS', 'CY3':'CYS', 
    'CYG':'CYS', 'CYM':'CYS', 'CYQ':'CYS', 'DAH':'PHE', 'DAL':'ALA', 'DAR':'ARG', 'DAS':'ASP', 'DCY':'CYS', 'DGL':'GLU', 'DGN':'GLN', 
    'DHA':'ALA', 'DHI':'HIS', 'DIL':'ILE', 'DIV':'VAL', 'DLE':'LEU', 'DLY':'LYS', 'DNP':'ALA', 'DPN':'PHE', 'DPR':'PRO', 'DSN':'SER', 
    'DSP':'ASP', 'DTH':'THR', 'DTR':'TRP', 'DTY':'TYR', 'DVA':'VAL', 'EFC':'CYS', 'FLA':'ALA', 'FME':'MET', 'GGL':'GLU', 'GL3':'GLY', 
    'GLZ':'GLY', 'GMA':'GLU', 'GSC':'GLY', 'HAC':'ALA', 'HAR':'ARG', 'HIC':'HIS', 'HIP':'HIS', 'HMR':'ARG', 'HPQ':'PHE', 'HTR':'TRP', 
    'HYP':'PRO', 'IAS':'ASP', 'IIL':'ILE', 'IYR':'TYR', 'KCX':'LYS', 'LLP':'LYS', 'LLY':'LYS', 'LTR':'TRP', 'LYM':'LYS', 'LYZ':'LYS', 
    'MAA':'ALA', 'MEN':'ASN', 'MHS':'HIS', 'MIS':'SER', 'MLE':'LEU', 'MPQ':'GLY', 'MSA':'GLY', 'MSE':'MET', 'MVA':'VAL', 'NEM':'HIS', 
    'NEP':'HIS', 'NLE':'LEU', 'NLN':'LEU', 'NLP':'LEU', 'NMC':'GLY', 'OAS':'SER', 'OCS':'CYS', 'OMT':'MET', 'PAQ':'TYR', 'PCA':'GLU', 
    'PEC':'CYS', 'PHI':'PHE', 'PHL':'PHE', 'PR3':'CYS', 'PRR':'ALA', 'PTR':'TYR', 'PYX':'CYS', 'SAC':'SER', 'SAR':'GLY', 'SCH':'CYS', 
    'SCS':'CYS', 'SCY':'CYS', 'SEL':'SER', 'SEP':'SER', 'SET':'SER', 'SHC':'CYS', 'SHR':'LYS', 'SMC':'CYS', 'SOC':'CYS', 'STY':'TYR', 
    'SVA':'SER', 'TIH':'ALA', 'TPL':'TRP', 'TPO':'THR', 'TPQ':'ALA', 'TRG':'LYS', 'TRO':'TRP', 'TYB':'TYR', 'TYI':'TYR', 'TYQ':'TYR', 
    'TYS':'TYR', 'TYY':'TYR', 'ALA':'ALA', 'ARG':'ARG', 'ASN':'ASN', 'ASP':'ASP', 'CYS':'CYS', 'GLU':'GLU', 'GLN':'GLN', 'GLY':'GLY', 
    'HIS':'HIS', 'ILE':'ILE', 'LEU':'LEU', 'LYS':'LYS', 'MET':'MET', 'PHE':'PHE', 'PRO':'PRO', 'SER':'SER', 'THR':'THR', 'TRP':'TRP', 
    'TYR':'TYR', 'VAL':'VAL', 'UNK':'UNK'
}


# Residue name to token
RESIDUE_NAME_TO_TOKEN = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLU": "E",
    "GLN": "Q",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    "UNK": "X",
}



# Allowed antigen types
ALLOWED_ANTIGEN_TYPES = {
    'protein',
    'protein | protein',
    'protein | protein | protein',
    'protein | protein | protein | protein',
    'protein | protein | protein | protein | protein',
}

# Using Chothia's antibody number system
# ==========================================================
CHOTHIA_RANGES = [
    "fr1", "cdr1", "fr2", "cdr2", "fr3", "cdr3", "fr4"
]

CDR_CHOTHIA = {
    'cdr1', 'cdr2', 'cdr3'
}
# ==========================================================


# Keep consistent resolution threshold with Boltz
RESOLUTION_THRESHOLD = 4.5


# We accitentally found that the file with the the following pdb id is wrong, so we need to remove it
# For example, when we visualize the structure of 4hjj, we found that the residue id is not correctly labeled. The labled variable region in the Chothia scheme does not bind the antigen.
# There might exist other wrong pdb id, but we only found 4hjj in the summary file for now.
WRONG_PDB_ID = {
    "4hjj"
}


# Convert nan to None
def nan_to_none(val):
    if val != val or not val:
        return None
    else:
        return val

# Convert nan to empty string
def nan_to_empty_string(val):
    if val != val or not val:
        return ''
    else:
        return val

# Split delimited string
def split_sabdab_delimited_str(val):
    if val != val or not val:       
        return []
    else:
        return sorted([s.strip() for s in val.split('|')])

# Parse SAbDab resolution
def parse_sabdab_resolution(val):
    if val == 'NOT' or not val or val != val:
        return None
    elif isinstance(val, str) and ',' in val:
        return float(val.split(',')[0].strip())
    else:
        return float(val)


# Check if the residue is an amino acid
def is_aa(value):
    return (value in STANDARD_RESIDUE_SUBSTITUTIONS_INCASEOF_NON_STANDARD_RESIDUE)


# Parse PDB data and return amino acid sequence. The truncation length is set to facilitate the subsequent processing of scFv antibodies, as abnumber cannot recognize the heavy chain of scfv antibodies.
def parse_biopython_structure(pdb_id, entity, max_seq_len = None):
    chains = Selection.unfold_entities(entity, 'C')  # C for chain
    chains.sort(key=lambda c: c.get_id())  # Sort chains by chain ID
    data = {}
    
    for _, chain in enumerate(chains):
        residues = Selection.unfold_entities(chain, 'R')  # R for residue
        residues.sort(key=lambda res: (res.get_id()[1], res.get_id()[2]))

        aa_sequence = []

        for index, res in enumerate(residues):

            """
            length truncation: 
            In chothia, 
            the VL region is the first 109 residues (including those with insertion codes)
            the VH region is the first 113 residues (including those with insertion codes)
            Reference: http://www.bioinf.org.uk/abs/info.html
            """
            res_id = int(res.get_id()[1])
            if max_seq_len is not None and res_id > max_seq_len:
                break

            resname = res.get_resname()
            if not is_aa(resname): 
                if resname != 'HOH':
                    logging.warning(f'PDB id: {pdb_id} Chain: {chain.get_id()} Index: {index + 1} Residue: {resname} Not an amino acid')
                continue

            aa_sequence.append(RESIDUE_NAME_TO_TOKEN[STANDARD_RESIDUE_SUBSTITUTIONS_INCASEOF_NON_STANDARD_RESIDUE[resname]])


        data[chain.get_id()] = ''.join(aa_sequence)

    return data


# abnumber can distinguish between heavy and light chains
# Used only for processing antibody sequences
def mask_cdr(seq, mask_token = 'X'):
    chain = abnumber.Chain(seq, scheme='chothia')
    masked = []
    origin_seq = []
    for range in CHOTHIA_RANGES:
        if len(getattr(chain, range + "_seq")) == 0: # check all CDRs/FRs are complete; or the CDR3 is too long, abnumber will return empty string for all regions
            return None, None
        origin_seq += list(getattr(chain, range + "_seq"))
        if range in CDR_CHOTHIA:
            masked += [mask_token] * len(getattr(chain, range + "_seq"))
        else:
            masked += list(getattr(chain, range + "_seq"))
    return ''.join(origin_seq), ''.join(masked)


def exchange_heavy_and_light_info_for_single_entry(wrong_entry_name, wrong_value_to_save):
    new_value_to_save = {
        'pdb': wrong_value_to_save['pdb'],
        'H_chain_id': wrong_value_to_save['L_chain_id'],
        'L_chain_id': wrong_value_to_save['H_chain_id'],
        'H_chain_seq': wrong_value_to_save['L_chain_seq'],
        'L_chain_seq': wrong_value_to_save['H_chain_seq'],
        'H_chain_masked_seq': wrong_value_to_save['L_chain_masked_seq'],
        'L_chain_masked_seq': wrong_value_to_save['H_chain_masked_seq'],
        'antigen_chain_id': wrong_value_to_save['antigen_chain_id'],
        'antigen_seq': wrong_value_to_save['antigen_seq'],
        'antigen_type': wrong_value_to_save['antigen_type'],
        'resolution': wrong_value_to_save['resolution'],
        'scfv': wrong_value_to_save['scfv'],
        'date': wrong_value_to_save['date'],
        'index_in_summary': wrong_value_to_save['index_in_summary']
    }
    new_entry_name = "{Pdb_id}_{H_chain}_{L_chain}_{Antigen_chains}".format(
        Pdb_id=wrong_value_to_save['pdb'],
        H_chain=nan_to_empty_string(wrong_value_to_save['L_chain_id']),
        L_chain=nan_to_empty_string(wrong_value_to_save['H_chain_id']),
        Antigen_chains=''.join(wrong_value_to_save['antigen_chain_id'])
    )
    assert len(new_entry_name) == len(wrong_entry_name)
    return new_entry_name, new_value_to_save


def check_single_entry(entry_name, value_to_save):
    if value_to_save['L_chain_id'] is None:  # skip the nanobody (only have heavy chain)
        return entry_name, value_to_save

    chain_heavy = abnumber.Chain(value_to_save['H_chain_seq'], scheme='chothia')
    chain_light = abnumber.Chain(value_to_save['L_chain_seq'], scheme='chothia')

    if chain_heavy.is_light_chain() and chain_light.is_heavy_chain():  # SabDab wrongly classify
        new_entry_name, new_value_to_save = exchange_heavy_and_light_info_for_single_entry(entry_name, value_to_save)
        return new_entry_name, new_value_to_save
    else:
        return entry_name, value_to_save

# We found that SabDab may wrongly classify the heavy chain as light chain, and vice versa
# Also check the chain type of the heavy chain and light chain are consistent with the chain type in the summary file
# This function is used to correct the classification
def correct_sabdab(json_content):
    results = Parallel(n_jobs=-1)(
        delayed(check_single_entry)(entry_name, value_to_save)
        for entry_name, value_to_save in tqdm(json_content.items(), desc='Checking SabDab classification', dynamic_ncols=True)
    )

    corrected_content = {}
    for _, (new_entry_name, new_value_to_save) in enumerate(results):
        corrected_content[new_entry_name] = new_value_to_save

    return corrected_content


def json_to_csv(json_content):
    file_name = []
    pdb = []
    H_chain_id = []
    L_chain_id = []
    H_chain_seq = []
    L_chain_seq = []
    H_chain_masked_seq = []
    L_chain_masked_seq = []
    antigen_chain_id = []
    antigen_seq = []
    antigen_type = []
    resolution = []
    scfv = []
    date = []
    index_in_summary = []
    
    for key, value in json_content.items():
        file_name.append(key)
        pdb.append(value["pdb"])
        H_chain_id.append(value["H_chain_id"])
        L_chain_id.append(value["L_chain_id"])
        H_chain_seq.append(value["H_chain_seq"])
        L_chain_seq.append(value["L_chain_seq"])
        H_chain_masked_seq.append(value["H_chain_masked_seq"])
        L_chain_masked_seq.append(value["L_chain_masked_seq"])
        antigen_chain_id.append(value["antigen_chain_id"])
        antigen_seq.append(value["antigen_seq"])
        antigen_type.append(value["antigen_type"])
        resolution.append(value["resolution"])
        scfv.append(value["scfv"])
        date.append(value["date"])
        index_in_summary.append(value["index_in_summary"])

    df = pd.DataFrame({
        "file_name": file_name,
        "pdb": pdb,
        "H_chain_id": H_chain_id,
        "L_chain_id": L_chain_id,
        "H_chain_seq": H_chain_seq,
        "L_chain_seq": L_chain_seq,
        "H_chain_masked_seq": H_chain_masked_seq,
        "L_chain_masked_seq": L_chain_masked_seq,
        "antigen_chain_id": antigen_chain_id,
        "antigen_seq": antigen_seq,
        "antigen_type": antigen_type,
        "resolution": resolution,
        "scfv": scfv,
        "date": date,
        "index_in_summary": index_in_summary
    })
    
    return df


def process_entry(row, index, chothia_dir, log_path, consider_no_antigen):
    """_summary_

    Args:
        row: a row in the summary tsv file
        index: index of the row in the summary file
        chothia_dir: chothia directory
        log_path: log file path
        consider_no_antigen: whether to consider the data with no antigen

    Returns:
        a tuple of (entry_name, value_to_save, A int)
        entry_name: the name of the entry
        value_to_save: the value to save, if failed to process, return None
        A int: 
            0 if the entry is processed, 
            1 if the entry is failed to read the pdb file (the pdb file is too large)
            2 if the entry is ignored
    """
    
    
    # Configure logging in each subprocess
    logging.basicConfig(
        filename=log_path,
        level=logging.WARNING,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filemode='a' # Append mode, won't cause the log file to be overwritten by each subprocess
    )
    
    entry_id = row['pdb'].lower()
    antigen_chains = split_sabdab_delimited_str(
        nan_to_empty_string(row['antigen_chain'])
    )  # may be multiple chains, a list
    entry_name = "{Pdb_id}_{H_chain}_{L_chain}_{Antigen_chains}".format(
        Pdb_id=entry_id,
        H_chain=nan_to_empty_string(row['Hchain']),
        L_chain=nan_to_empty_string(row['Lchain']),
        Antigen_chains=''.join(antigen_chains)
    )

    entry_ag_type = nan_to_none(row['antigen_type'])
    entry_resolution = parse_sabdab_resolution(row['resolution'])
    entry_is_scfv = row['scfv']
    entry_date = row['date']

    value_to_save = {
        'pdb': entry_id,
        'H_chain_id': None,
        'L_chain_id': None,
        'H_chain_seq': None,
        'L_chain_seq': None,
        'H_chain_masked_seq': None,
        'L_chain_masked_seq': None,
        'antigen_chain_id': antigen_chains,
        'antigen_seq': None,
        'antigen_type': entry_ag_type,
        'resolution': entry_resolution,
        'scfv': entry_is_scfv,
        'date': entry_date,
        'index_in_summary': index
    }
    
    if entry_id in WRONG_PDB_ID:
        return entry_name, None, 2

    
    if consider_no_antigen:
        if not (
                (entry_ag_type in ALLOWED_ANTIGEN_TYPES or entry_ag_type is None)
                and (entry_resolution is not None and entry_resolution <= RESOLUTION_THRESHOLD)
            ):
            return entry_name, None, 2
    else:
        if not (
                (entry_ag_type in ALLOWED_ANTIGEN_TYPES)
                and (entry_resolution is not None and entry_resolution <= RESOLUTION_THRESHOLD)
            ):
            return entry_name, None, 2

    try:
        entry_pdb_path = os.path.join(chothia_dir, f'{entry_id}.pdb')
        pdb_parser = PDB.PDBParser(QUIET=True)
        entry_structure = pdb_parser.get_structure(entry_id, entry_pdb_path)[0]
    except Exception as e:
        error_message = f'Summary Index: {index}, PDB id: {entry_id} - Error loading structure: {e}'
        logging.error(error_message)
        return entry_name, None, 1  # None: ignore this sample

    
    # H chain
    if nan_to_none(row['Hchain']) is not None:
        try:
            value_to_save['H_chain_id'] = row['Hchain']
            raw_H_seq = parse_biopython_structure(entry_id, entry_structure[row['Hchain']], max_seq_len=113)[row['Hchain']]
            value_to_save['H_chain_seq'], value_to_save['H_chain_masked_seq'] = mask_cdr(raw_H_seq)
            if value_to_save['H_chain_seq'] is None:
                logging.error(f'{entry_name}: Heavy Chain {row["Hchain"]} CDRs/FRs are not complete or too long CDR3, Seq: {raw_H_seq}')
                return entry_name, None, 2
        except Exception as e:
            error_message = f'{entry_name}: Heavy chain error for chain named {row["Hchain"]}: {e}'
            logging.error(error_message)
            return entry_name, None, 2  # None: ignore this sample
    else:
        return entry_name, None, 2 # abandon data with no heavy chain

    # L chain
    if nan_to_none(row['Lchain']) is not None:
        try:
            value_to_save['L_chain_id'] = row['Lchain']
            raw_L_seq = parse_biopython_structure(entry_id, entry_structure[row['Lchain']], max_seq_len=106)[row['Lchain']]
            value_to_save['L_chain_seq'], value_to_save['L_chain_masked_seq'] = mask_cdr(raw_L_seq)
            if value_to_save['L_chain_seq'] is None:
                logging.error(f'{entry_name}: Light Chain {row["Lchain"]} CDRs/FRs are not complete, Seq: {raw_L_seq}')
                return entry_name, None, 2
        except Exception as e:
            error_message = f'{entry_name}: Light chain error for chain named {row["Lchain"]}: {e}'
            logging.error(error_message)
            return entry_name, None, 2  # None: ignore this sample

    # Antigen
    antigen_chains_to_process = [entry_structure[c] for c in antigen_chains]
    value_to_save['antigen_seq'] = parse_biopython_structure(entry_id, antigen_chains_to_process)

    return entry_name, value_to_save, 0


# Process raw data
def process_raw_data(args):

    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(args.output_dir, 'logs-for-preprocess')
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir,  f'processing_{current_time}.log')

    chothia_dir = args.chothia_dir
    summary_dir = args.summary_dir
    output_dir = args.output_dir
    consider_no_antigen_flag = args.consider_no_antigen

    summary_df = pd.read_csv(summary_dir, sep='\t')

    processed_data = {}

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


    # using joblib to parallelize the processing
    results = Parallel(n_jobs=-1)(
        delayed(process_entry)(row, index, chothia_dir, log_path, consider_no_antigen_flag) for index, row in tqdm(
            summary_df.iterrows(), 
            total=len(summary_df), 
            desc='Processing raw data', 
            dynamic_ncols=True
        )
    )
    
    failed_to_process_with_bio = []

    for entry_name, value_to_save, indicator in results:
        if value_to_save is not None:
            processed_data[entry_name] = value_to_save
        else:
            if indicator == 1:
                failed_to_process_with_bio.append(entry_name)
                
    processed_data = correct_sabdab(processed_data) # SabDab may wrongly classify the heavy chain as light chain, and vice versa
    
    df = json_to_csv(processed_data)
    df.to_csv(os.path.join(output_dir, os.path.splitext(args.output_file_name)[0] + '.csv'), index=False)
    
    with open(os.path.join(output_dir, 'failed_to_process_with_bio.json'), 'w') as f:
        json.dump(failed_to_process_with_bio, f, indent=4)

    with open(os.path.join(output_dir, args.output_file_name), 'w') as f:
        json.dump(processed_data, f, indent=4)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--chothia_dir', type=str, default='./data/raw_data/chothia/')
    parser.add_argument('--summary_dir', type=str, default='./data/raw_data/sabdab_summary_all.tsv')
    parser.add_argument('--output_dir', type=str, default='./data/')
    parser.add_argument('--output_file_name', type=str, default='summary.json')
    parser.add_argument('--consider_no_antigen', action='store_true')
    args = parser.parse_args()
    process_raw_data(args)
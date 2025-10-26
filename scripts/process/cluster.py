import os
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import subprocess
import argparse
import shutil

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


def processed_csv_to_fasta_only_cdr3(processed_csv_path, out_dir):
    data = pd.read_csv(processed_csv_path)
    
    
    out_fasta_path_for_heavy_cdr1 = os.path.join(out_dir, f"heavy_chain_cdr1.fasta")
    out_fasta_path_for_heavy_cdr2 = os.path.join(out_dir, f"heavy_chain_cdr2.fasta")
    out_fasta_path_for_heavy_cdr3 = os.path.join(out_dir, f"heavy_chain_cdr3.fasta")
    
    out_fasta_path_for_light_cdr1 = os.path.join(out_dir, f"light_chain_cdr1.fasta")
    out_fasta_path_for_light_cdr2 = os.path.join(out_dir, f"light_chain_cdr2.fasta")
    out_fasta_path_for_light_cdr3 = os.path.join(out_dir, f"light_chain_cdr3.fasta")
    
    
    out_fasta_paths = [out_fasta_path_for_heavy_cdr1, 
                        out_fasta_path_for_heavy_cdr2, 
                        out_fasta_path_for_heavy_cdr3, 
                        out_fasta_path_for_light_cdr1, 
                        out_fasta_path_for_light_cdr2, 
                        out_fasta_path_for_light_cdr3]
    
    for fasta_path in out_fasta_paths:
        if os.path.exists(fasta_path):
            print(f"The file {fasta_path} already exists, remove it and will re-generate it.")
            os.remove(fasta_path)
    
    
    print(f"Next, we will write heavy chain fasta to:\n{out_fasta_path_for_heavy_cdr1}\n{out_fasta_path_for_heavy_cdr2}\n{out_fasta_path_for_heavy_cdr3}")
    print(f"Next, we will write light chain fasta to:\n{out_fasta_path_for_light_cdr1}\n{out_fasta_path_for_light_cdr2}\n{out_fasta_path_for_light_cdr3}")
    print("Begin to generate fasta files...")
    

    out_fasta_path_for_heavy_cdr1_content = []
    out_fasta_path_for_heavy_cdr2_content = []
    out_fasta_path_for_heavy_cdr3_content = []
    
    for _, row in tqdm(data.iterrows(), desc='Writing heavy chain fasta'):
        if not pd.isna(row['H_chain_seq']):
            cdr1, cdr2, cdr3 = get_cdr(row['H_chain_seq'], row['H_chain_masked_seq'])
            out_fasta_path_for_heavy_cdr1_content.append(f">{row['file_name']} mol:protein length:{len(cdr1)} Protein\n{cdr1}")
            out_fasta_path_for_heavy_cdr2_content.append(f">{row['file_name']} mol:protein length:{len(cdr2)} Protein\n{cdr2}")
            out_fasta_path_for_heavy_cdr3_content.append(f">{row['file_name']} mol:protein length:{len(cdr3)} Protein\n{cdr3}")

    with open(out_fasta_path_for_heavy_cdr1, 'w') as f_cdr1:
        f_cdr1.write("\n".join(out_fasta_path_for_heavy_cdr1_content))
    with open(out_fasta_path_for_heavy_cdr2, 'w') as f_cdr2:
        f_cdr2.write("\n".join(out_fasta_path_for_heavy_cdr2_content))
    with open(out_fasta_path_for_heavy_cdr3, 'w') as f_cdr3:
        f_cdr3.write("\n".join(out_fasta_path_for_heavy_cdr3_content))
    
    out_fasta_path_for_light_cdr1_content = []
    out_fasta_path_for_light_cdr2_content = []
    out_fasta_path_for_light_cdr3_content = []
    for _, row in tqdm(data.iterrows(), desc='Writing light chain fasta'):
        if not pd.isna(row['L_chain_seq']):
            cdr1, cdr2, cdr3 = get_cdr(row['L_chain_seq'], row['L_chain_masked_seq'])
            out_fasta_path_for_light_cdr1_content.append(f">{row['file_name']} mol:protein length:{len(cdr1)} Protein\n{cdr1}")
            out_fasta_path_for_light_cdr2_content.append(f">{row['file_name']} mol:protein length:{len(cdr2)} Protein\n{cdr2}")
            out_fasta_path_for_light_cdr3_content.append(f">{row['file_name']} mol:protein length:{len(cdr3)} Protein\n{cdr3}")
            
    with open(out_fasta_path_for_light_cdr1, 'w') as f_cdr1:
        f_cdr1.write("\n".join(out_fasta_path_for_light_cdr1_content))
    with open(out_fasta_path_for_light_cdr2, 'w') as f_cdr2:
        f_cdr2.write("\n".join(out_fasta_path_for_light_cdr2_content))
    with open(out_fasta_path_for_light_cdr3, 'w') as f_cdr3:
        f_cdr3.write("\n".join(out_fasta_path_for_light_cdr3_content))
                
    print("Done!")
                
    return out_fasta_paths
                





def use_mmseqs_to_cluster(mmseqs_path, fasta_path, cluster_dir, seq_identity_threshold):
    print(f"Begin to cluster {fasta_path} with mmseqs...")
    fasta_PATH = Path(fasta_path)
    region_name = fasta_PATH.stem
    sub_cluster_dir = cluster_dir / region_name
    if sub_cluster_dir.exists():
        print(f"The sub cluster directory {sub_cluster_dir} already exists, remove it and will conduct re-clustering.")
        shutil.rmtree(sub_cluster_dir)
    sub_cluster_dir.mkdir(parents=True)
    
    subprocess.run(
        f"{mmseqs_path} easy-cluster {fasta_PATH} {sub_cluster_dir}/cluster_{region_name} {sub_cluster_dir}/tmp_{region_name} --min-seq-id {seq_identity_threshold}",
        shell=True,
        check=True,
    )

    print(f"Done! The result is in {sub_cluster_dir}/clust_{region_name}_similarity_{seq_identity_threshold}.tsv")
    
    
    


def main(args):
    
    out_dir = args.out_dir
    processed_data_path = Path(args.processed_csv_path)
    mmseqs_path = args.mmseqs_path
    seq_identity_threshold = args.seq_identity_threshold
    
    decuplication_threshold = args.decuplication_threshold
    decuplication_threshold_of_specified_processed_data = processed_data_path.stem.split('_')[-1]
    
    if decuplication_threshold_of_specified_processed_data != decuplication_threshold:
        print(f"The decuplication threshold of the specified processed data is {decuplication_threshold_of_specified_processed_data}, but the decuplication threshold is {decuplication_threshold}.")
        parts = processed_data_path.stem.split('_')
        parts[-1] = str(decuplication_threshold) 
        new_stem = '_'.join(parts)
        processed_data_path = processed_data_path.with_name(f"{new_stem}.csv")
        print(f"The processed data path is now {processed_data_path}.")

    
    out_dir = Path(os.path.join(out_dir, f"decuplicated_threshold_{decuplication_threshold}"))
    out_dir.mkdir(parents=True, exist_ok=True)

    generated_fasta_paths = processed_csv_to_fasta_only_cdr3(processed_data_path, out_dir)
    
    cluster_dir = Path(out_dir, f"cluster_results_seq_identity_threshold_{seq_identity_threshold}")
    if cluster_dir.exists():
        print(f"The cluster directory {cluster_dir} already exists, remove it and will conduct re-clustering.")
        shutil.rmtree(cluster_dir)
    cluster_dir.mkdir(parents=True) 
    
    for fasta_path in generated_fasta_paths:
        use_mmseqs_to_cluster(mmseqs_path, fasta_path, cluster_dir, seq_identity_threshold)

    
if __name__ == "__main__":
    
    # We found that "MMseqs2 Release 15-6f452" works well for short sequences (CDRs). In the beginning, we tried "MMseqs2 Release 16-747c6", but it failed but we think the developers will fix it in the future.
    # https://github.com/soedinglab/MMseqs2/releases/tag/15-6f452

    args = argparse.ArgumentParser()
    args.add_argument("--processed_csv_path", type=str, default="./data/summary-decuplication-distance_threshold_9.csv")
    args.add_argument("--decuplication_threshold", type=int, default=9)
    args.add_argument("--out_dir", type=str, default="./data")
    args.add_argument("--seq_identity_threshold", type=float, default=0.5)
    args.add_argument("--mmseqs_path", type=str, default="./mmseqs/bin/mmseqs")
    args = args.parse_args()
    
    main(args)

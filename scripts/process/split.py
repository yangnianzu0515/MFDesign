import pandas as pd
import numpy as np
import os
import contextlib
from tqdm import tqdm
import json
import argparse
from collections import defaultdict

# Following DiffAb and AbDiffuser, we also split the data into train, validation and test sets based on the CDR-H3 sequence identity.

def main(args):
    
    clu_result_path = args.clu_result_path # cluster result path via mmseqs2
    before_cutoff_in_sabdab_path = args.pdb_before_cutoff_in_sabdab_path # pdb before cut-off date (2021-9-30) in sabdab

    # load pdb before cut-off date (2021-9-30) in sabdab
    with open(before_cutoff_in_sabdab_path, "r") as f:
        before_cutoff_in_sabdab = json.load(f)

    cluster_tsv = pd.read_csv(clu_result_path, sep="\t", header=None) # load cluster result
    cluster_id = cluster_tsv[0] # cluster id
    entry = cluster_tsv[1]
    clustering_result = dict(zip(list(entry), list(cluster_id))) # dict: key is entry, value is cluster_id
    
    pdb_for_train = set(before_cutoff_in_sabdab) # pdb which we must reverse in the train set based on the cut-off date
    for key, value in clustering_result.items(): # iterate over the clustering result
        pdb_id = key[:4] # pdb id of the entry
        cluster_pdb = value[:4] # cluster pdb id
        
        # find those pdbs whose cluster is exactly before cut-off date
        if cluster_pdb in before_cutoff_in_sabdab:
            pdb_for_train.add(pdb_id)
            
        # find those clusters which includes pdbs before cut-off date
        if pdb_id in before_cutoff_in_sabdab:
            pdb_for_train.add(cluster_pdb)

    
    # find those clusters which includes pdbs before cut-off date. In our case, we find that the current cluster_for_train is less than 0.8*total_size. So we need to add more clusters to the train set in the next step.
    cluster_for_train = set()
    for key, value in clustering_result.items():
        if key[:4] in pdb_for_train:
            cluster_for_train.add(value[:4])


    all_clusters = set([item[:4] for item in clustering_result.values()])
    total_size = len(all_clusters)
    train_size = int(total_size * 0.9)
    val_size = int(total_size * 0.05)
    test_size = total_size - train_size - val_size
    
    assert train_size + val_size + test_size == total_size

    # find those clusters which are not in the cluster_for_train set and we will split them into train, val and test sets further
    remaining_clusters = all_clusters - cluster_for_train
    
    # In the next, we will sort the remaining clusters based on the string order of the cluster id.
    remaining_clusters = sorted(remaining_clusters)
    
    initial_train_size = len(cluster_for_train)
    cluster_for_train = cluster_for_train.union(set(remaining_clusters[:train_size - initial_train_size]))
    

    # if we directly assign the remaining clusters to val and test sets by the index in the remaining_clusters, the size of val and test sets will be very different
    # so we adopt a easy way: alternatively assign the remaining clusters to val and test sets to let the size of val and test sets be close to each other
    cluster_for_val = set()
    cluster_for_test = set()
    
    for i, cluster in enumerate(remaining_clusters[train_size - initial_train_size:]):
        if i % 2 == 0 and len(cluster_for_val) < val_size:
            cluster_for_val.add(cluster)
        elif len(cluster_for_test) < test_size:
            cluster_for_test.add(cluster)
    
    # We count the number of pdbs in train/val/test clusters
    train_entry = []
    val_entry = []
    test_entry = []
    for key, value in clustering_result.items():
        if value[:4] in cluster_for_train:
            train_entry.append(key)
        elif value[:4] in cluster_for_val:
            val_entry.append(key)
        elif value[:4] in cluster_for_test:
            test_entry.append(key)
    
    # if the size of val set is larger than test set, we swap the size of val and test sets
    if len(val_entry) > len(test_entry):
        val_size, test_size = test_size, val_size
        val_entry, test_entry = test_entry, val_entry
    
    print(f"total size: {total_size} clusters")
    print(f"train size: {train_size} clusters")
    print(f"val size: {val_size} clusters")
    print(f"test size: {test_size} clusters")
    print("="*50)
    
    print(f"train entry count: {len(train_entry)}")
    print(f"val entry count: {len(val_entry)}")
    print(f"test entry count: {len(test_entry)}")


    # save the train, val and test sets
    with open(os.path.join(args.output_dir, "train_entry.json"), "w") as f:
        json.dump(train_entry, f)
    with open(os.path.join(args.output_dir, "val_entry.json"), "w") as f:
        json.dump(val_entry, f)
    with open(os.path.join(args.output_dir, "test_entry.json"), "w") as f:
        json.dump(test_entry, f)
    print("Save the train, val and test sets successfully to the output dir.")

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--clu_result_path", type=str, default="./data/decuplicated_threshold_9/cluster_results_seq_identity_threshold_0.5/heavy_chain_cdr3/cluster_heavy_chain_cdr3_cluster.tsv")
    args.add_argument("--pdb_before_cutoff_in_sabdab_path", type=str, default="./data/before_cutoff_in_sabdab.json")
    args.add_argument("--precessed_data_path", type=str, default="./data/summary-decuplication-distance_threshold_9.csv")
    args.add_argument("--output_dir", type=str, default="./data")
    args = args.parse_args()
    main(args)
    
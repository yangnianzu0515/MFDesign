import datetime
import json
import pandas as pd
import gemmi
import contextlib
import os
from tqdm import tqdm
import argparse


# copied from Boltz repository
# https://github.com/jwohlwend/boltz
def get_dates(block: gemmi.cif.Block) -> tuple[str, str, str]:
    """Get the deposited, released, and last revision dates.

    Parameters
    ----------
    block : gemmi.cif.Block
        The block to process.

    Returns
    -------
    str
        The deposited date.
    str
        The released date.
    str
        The last revision date.

    """
    deposited = "_pdbx_database_status.recvd_initial_deposition_date"
    revision = "_pdbx_audit_revision_history.revision_date"
    deposit_date = revision_date = release_date = ""
    with contextlib.suppress(Exception):
        deposit_date = block.find([deposited])[0][0]
        release_date = block.find([revision])[0][0]
        revision_date = block.find([revision])[-1][0]

    return deposit_date, release_date, revision_date

def get_pdb_id_before_cutoff_date(summary: pd.DataFrame, cutoff_date: datetime.datetime, cif_dir: str) -> tuple[set[str], set[str]]:
    before_cutoff_in_sabdab = set()
    before_cutoff_in_boltz = set()
    for i in tqdm(range(len(summary)), desc="Checking dates"):
        row = summary.iloc[i]
        pbd_id = row["pdb"]
        date_in_summary = datetime.datetime.strptime(row['date'], '%m/%d/%y')
        cif_path = os.path.join(cif_dir, f"{pbd_id}.cif")
        block = gemmi.cif.read(str(cif_path))[0]
        deposit_date, release_date, revision_date = get_dates(block)
        release_date = datetime.datetime.strptime(release_date, '%Y-%m-%d') # release date is used for boltz when filtering training data
        
        if date_in_summary <= cutoff_date:
            before_cutoff_in_sabdab.add(pbd_id)
        if release_date <= cutoff_date:
            before_cutoff_in_boltz.add(pbd_id)
    return before_cutoff_in_sabdab, before_cutoff_in_boltz


def main(args):
    summary = pd.read_csv(args.summary_file_path)
    cutoff_date = datetime.datetime.strptime(args.cutoff_date, '%Y-%m-%d')
    before_cutoff_in_sabdab, before_cutoff_in_boltz = get_pdb_id_before_cutoff_date(summary, cutoff_date, args.cif_dir)
    
    # check whether entries before cutoff date in boltz are also in sabdab
    # if so, we can directly split data based on the date provided in sabdab
    if before_cutoff_in_boltz.issubset(before_cutoff_in_sabdab):
        print("Entries before cutoff date in boltz are also all in sabdab")
    else:
        print("Entries before cutoff date in boltz are not all in sabdab")
        
    
    with open(os.path.join(args.out_dir, "before_cutoff_in_sabdab.json"), "w") as f:
        json.dump(list(before_cutoff_in_sabdab), f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary_file_path", type=str, default="./data/summary-decuplication-distance_threshold_9.csv")
    parser.add_argument("--cif_dir", type=str, default="./data/raw_data/cif")
    parser.add_argument("--out_dir", type=str, default="./data")
    parser.add_argument("--cutoff_date", type=str, default="2021-09-30")
    args = parser.parse_args()
    main(args)

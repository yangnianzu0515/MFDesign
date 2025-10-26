# From https://github.com/sokrypton/ColabFold/blob/main/colabfold/colabfold.py

import logging
import os
import random
import tarfile
import time
from typing import Union

import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)

TQDM_BAR_FORMAT = (
    "{l_bar}{bar}| {n_fmt}/{total_fmt} [elapsed: {elapsed} remaining: {remaining}]"
)


def run_mmseqs2(  # noqa: PLR0912, D103, C901, PLR0915
    x: Union[str, list[str]],
    prefix: str = "tmp",
    use_env: bool = True,
    use_filter: bool = True,
    use_pairing: bool = False,
    pairing_strategy: str = "greedy",
    host_url: str = "https://api.colabfold.com",
) -> tuple[list[str], list[str]]:
    """
    x: a list of sequences strings
    """
    submission_endpoint = "ticket/pair" if use_pairing else "ticket/msa"

    # Set header agent as boltz
    headers = {}
    headers["User-Agent"] = "boltz"

    """
    submit: submit the job to the mmseqs2 api
    """
    def submit(seqs, mode, N=101):
        n, query = N, ""
        for seq in seqs:
            query += f">{n}\n{seq}\n"
            n += 1
        """
        query is like this:
        >101
        seq1
        >102
        seq2
        ...
        """    
            

        while True:
            error_count = 0
            # initialize the error count to 0
            try:
                # https://requests.readthedocs.io/en/latest/user/advanced/#advanced
                # "good practice to set connect timeouts to slightly larger than a multiple of 3"
                res = requests.post(
                    f"{host_url}/{submission_endpoint}",
                    data={"q": query, "mode": mode},
                    timeout=6.02,
                    headers=headers,
                )
            except requests.exceptions.Timeout:
                # if the exception is a timeout, inform the user and continue
                logger.warning("Timeout while submitting to MSA server. Retrying...")
                continue
                # continue: means try again
            except Exception as e:
                # if the exception is not a timeout, increment the error count and inform the user
                error_count += 1
                logger.warning(
                    f"Error while fetching result from MSA server. Retrying... ({error_count}/5)"
                )
                logger.warning(f"Error: {e}")
                time.sleep(5)
                if error_count > 5:
                    raise
                    # if there have been more than 5 errors, raise an exception, and stop the submit function
                continue
            break
            # if the request is successful, break the while loop;

        try:
            out = res.json()
        except ValueError:
            logger.error(f"Server didn't reply with json: {res.text}")
            out = {"status": "ERROR"}
        return out

    # status: get the status of the job
    def status(ID):
        while True:
            error_count = 0
            try:
                res = requests.get(
                    f"{host_url}/ticket/{ID}", timeout=6.02, headers=headers
                )
            except requests.exceptions.Timeout:
                logger.warning(
                    "Timeout while fetching status from MSA server. Retrying..."
                )
                continue
            except Exception as e:
                error_count += 1
                logger.warning(
                    f"Error while fetching result from MSA server. Retrying... ({error_count}/5)"
                )
                logger.warning(f"Error: {e}")
                time.sleep(5)
                if error_count > 5:
                    raise
                continue
            break
        try:
            out = res.json()
        except ValueError:
            logger.error(f"Server didn't reply with json: {res.text}")
            out = {"status": "ERROR"}
        return out

    # download: download the result of the job, and save it to the given path
    def download(ID, path):
        error_count = 0
        while True:
            try:
                res = requests.get(
                    f"{host_url}/result/download/{ID}", timeout=6.02, headers=headers
                )
            except requests.exceptions.Timeout:
                logger.warning(
                    "Timeout while fetching result from MSA server. Retrying..."
                )
                continue
            except Exception as e:
                error_count += 1
                logger.warning(
                    f"Error while fetching result from MSA server. Retrying... ({error_count}/5)"
                )
                logger.warning(f"Error: {e}")
                time.sleep(5)
                if error_count > 5:
                    raise
                continue
            break
        with open(path, "wb") as out:
            out.write(res.content)

    # process input x
    seqs = [x] if isinstance(x, str) else x
    # in our ab design, x is always a list of sequences strings, so seqs is just x
    

    # setup mode
    if use_filter:
        mode = "env" if use_env else "all"
    else:
        mode = "env-nofilter" if use_env else "nofilter"


    if use_pairing:
        mode = ""
        # greedy is default, complete was the previous behavior
        if pairing_strategy == "greedy":
            mode = "pairgreedy"
        elif pairing_strategy == "complete":
            mode = "paircomplete"
        if use_env:
            mode = mode + "-env"

    # define path
    path = f"{prefix}_{mode}"
    if not os.path.isdir(path):
        os.mkdir(path)

    # call mmseqs2 api
    tar_gz_file = f"{path}/out.tar.gz"
    N, REDO = 101, True

    # deduplicate and keep track of order
    seqs_unique = []
    # TODO this might be slow for large sets
    [seqs_unique.append(x) for x in seqs if x not in seqs_unique]
    Ms = [N + seqs_unique.index(seq) for seq in seqs]
    # Ms = [101, 102, 103, ..., 101+len(seqs_unique)-1]
    # lets do it!
    if not os.path.isfile(tar_gz_file):
    # if the tar_gz_file does not exist, we need to submit the job to the mmseqs2 api
        TIME_ESTIMATE = 150 * len(seqs_unique)
        # 
        with tqdm(total=TIME_ESTIMATE, bar_format=TQDM_BAR_FORMAT) as pbar:
            while REDO:
                # continue to submit the job until it goes through, i.e. REDO is set to False
                # set the description of the progress bar to SUBMIT
                pbar.set_description("SUBMIT")
                
                # Resubmit job until it goes through
                out = submit(seqs_unique, mode, N)
                while out["status"] in ["UNKNOWN", "RATELIMIT"]:
                    sleep_time = 5 + random.randint(0, 5)
                    logger.error(f"Sleeping for {sleep_time}s. Reason: {out['status']}")
                    # resubmit
                    time.sleep(sleep_time)
                    out = submit(seqs_unique, mode, N)
                    # if the status is UNKNOWN or RATELIMIT, we need to resubmit the job until the status is not UNKNOWN or RATELIMIT

                # if the status is ERROR, we need to raise an exception
                if out["status"] == "ERROR":
                    msg = (
                        "MMseqs2 API is giving errors. Please confirm your "
                        " input is a valid protein sequence. If error persists, "
                        "please try again an hour later."
                    )
                    raise Exception(msg)

                # if the status is MAINTENANCE, we need to raise an exception
                if out["status"] == "MAINTENANCE":
                    msg = (
                        "MMseqs2 API is undergoing maintenance. "
                        "Please try again in a few minutes."
                    )
                    raise Exception(msg)

                # wait for job to finish
                ID, TIME = out["id"], 0
                # ID is the job id
                pbar.set_description(out["status"])
                while out["status"] in ["UNKNOWN", "RUNNING", "PENDING"]:
                    t = 5 + random.randint(0, 5)
                    logger.error(f"Sleeping for {t}s. Reason: {out['status']}")
                    time.sleep(t)
                    # sleep for t seconds to wait for the job to finish
                    out = status(ID)
                    pbar.set_description(out["status"])
                    if out["status"] == "RUNNING":
                        TIME += t
                        pbar.update(n=t)
                # if the status is not UNKNOWN, RUNNING, or PENDING, break the while loop

                # if the status is COMPLETE, we need to update the progress bar and set REDO to False
                if out["status"] == "COMPLETE":
                    if TIME < TIME_ESTIMATE:
                        pbar.update(n=(TIME_ESTIMATE - TIME))
                    REDO = False
                    
                # if the status is ERROR, we need to raise an exception
                if out["status"] == "ERROR":
                    REDO = False
                    msg = (
                        "MMseqs2 API is giving errors. Please confirm your "
                        " input is a valid protein sequence. If error persists, "
                        "please try again an hour later."
                    )
                    raise Exception(msg)

            # Download results
            download(ID, tar_gz_file)

    # prep list of a3m files
    if use_pairing:
        a3m_files = [f"{path}/pair.a3m"]
    else:
        a3m_files = [f"{path}/uniref.a3m"]
        if use_env:
        # use_env is always True
            a3m_files.append(f"{path}/bfd.mgnify30.metaeuk30.smag30.a3m")

    # extract a3m files
    if any(not os.path.isfile(a3m_file) for a3m_file in a3m_files):
        # if any of the a3m files do not exist, we need to extract the tar_gz_file
        with tarfile.open(tar_gz_file) as tar_gz:
            tar_gz.extractall(path)

    # gather a3m lines
    a3m_lines = {}
    for a3m_file in a3m_files:
        update_M, M = True, None
        for line in open(a3m_file, "r"):
            if len(line) > 0:
                if "\x00" in line:
                    # if the line contains "\x00", it means that it comes to the results of a new sequence
                    line = line.replace("\x00", "")
                    update_M = True
                if line.startswith(">") and update_M:
                    M = int(line[1:].rstrip())
                    # M is the sequence id in the a3m file
                    update_M = False
                    if M not in a3m_lines:
                        a3m_lines[M] = []
                a3m_lines[M].append(line)

    a3m_lines = ["".join(a3m_lines[n]) for n in Ms]
    """
    if use_pairing is True, the format of a3mlines is like this:
    [
        ">101\nseq1\n>UniRef100_Q9Y6K8\nseq2\n>UniRef100_222228\nseq3\n",
        ">102\nseq1\n>UniRef100_123328\nseq2\n>UniRef100_22SDS3\nseq3\n",
        ">103\nseq1\n>UniRef100_9234K2\nseq2\n>UniRef100_22FSF2\nseq3\n",
        ...
    ]
    
    seq may be "DUMMY"
    
    
    if use_pairing is False, the format of a3mlines is like this:
    [
        ">101\nseq1\n>UniRef100_Q9Y6K8\nseq2\n>UniRef100_222228\nseq3\n.....>101 again .....",
        ">102\nseq1\n>UniRef100_123328\nseq2\n>UniRef100_22SDS3\nseq3\n.....>102 again .....",
        ">103\nseq1\n>UniRef100_9234K2\nseq2\n>UniRef100_22FSF2\nseq3\n.....>103 again .....",
        ...
    ]
    """
    
    
    return a3m_lines

import argparse
import json
import sys
import os
import concurrent.futures
from pathlib import Path
from tqdm import tqdm
import numpy as np

# --- Prerequisites ---
# Make sure the 'src' directory containing the 'boltz' library
# is in the Python Path.
sys.path.insert(0, './src')

try:
    from boltz.data.parse.csv import parse_csv_for_ab_design
    from boltz.data.types import MSA
except ImportError as e:
    print("Error: Could not import the 'boltz' library.")
    print("Please make sure you are running this script from the project's root directory.")
    print(f"Error details: {e}")
    sys.exit(1)

def derive_target_id_from_path(csv_path: Path) -> str:
    """
    Derives the target_id from the CSV filename.
    Assumes the format: {target_id}_{entity_id}.csv
    """
    stem = csv_path.stem
    parts = stem.split('_')
    target_id = '_'.join(parts[:-1])
    
    if not target_id:
        print(f"Warning: Could not derive a valid target_id from file '{csv_path.name}'. "
              f"Using the full stem '{stem}' as ID.")
        return stem
        
    return target_id

def process_single_msa(
    msa_csv_path: Path,
    output_dir: Path,
    msa_filtering_threshold: float,
    preprocessed_data: dict,
    max_seqs: int,
) -> str:
    """
    Worker function that processes a single MSA file.
    Returns a status message indicating success or failure.
    """
    try:
        # 1. Automatically derive the target_id from the filename
        target_id = derive_target_id_from_path(msa_csv_path)

        # 2. Get the 'entry_info' for the specific target_id
        # This check is now guaranteed by the pre-filtering in main(), but kept for safety.
        entry_info = preprocessed_data[target_id]

        # 3. Call the parsing function to process the MSA
        msa, num_before, num_after = parse_csv_for_ab_design(
            path=msa_csv_path,
            max_seqs=max_seqs,
            entry_info=entry_info,
            msa_filtering_threshold=msa_filtering_threshold,
        )

        # 4. Construct the output path and save the .npz file
        output_npz_path = output_dir / f"{msa_csv_path.stem}.npz"
        msa.dump(output_npz_path)

        return { 
            "status": "success",
            "file": msa_csv_path.name,
            "before": num_before,
            "after": num_after
        }

    except KeyError:
        return { 
            "status": "failure",
            "message": f"Failure (KeyError): ID '{target_id}' not found in JSON for '{msa_csv_path.name}'."
        }
    except Exception as e:
        return { 
            "status": "failure",
            "message": f"Failure (Exception): Error processing '{msa_csv_path.name}': {e}"
        }


def main():
    parser = argparse.ArgumentParser(
        description="Batch process and parallelize the conversion of MSA files from CSV to NPZ format for Boltz.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing the input MSA files in .csv format.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory where the output .npz files will be saved.")
    parser.add_argument("--msa_filtering_threshold", type=float, default=0.2, help="Threshold for MSA filtering.")
    parser.add_argument("--preprocessed_data_path", type=str, default="./data/summary.json", help="Path to the metadata JSON file (e.g., summary.json).")
    
    parser.add_argument("--max_seqs", type=int, default=4096, help="Maximum number of sequences to process from the MSA.")
    parser.add_argument("--num_workers", type=int, default=os.cpu_count(), help="Number of parallel processes to use. Defaults to all available CPU cores.")

    args = parser.parse_args()

    # --- Preparation ---
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load the preprocessed data once to be shared by all processes
    print(f"Loading preprocessed data from: {args.preprocessed_data_path}")
    try:
        with open(args.preprocessed_data_path, "r") as f:
            preprocessed_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Preprocessed data file not found at '{args.preprocessed_data_path}'")
        sys.exit(1)

    # Find all CSV files in the input directory
    all_csv_files = list(input_dir.glob("*.csv"))
    if not all_csv_files:
        print(f"No .csv files found in directory '{input_dir}'.")
        return

    print(f"Found {len(all_csv_files)} total CSV files in the directory.")

    # --- NEW: Filter the list of files based on the preprocessed_data JSON ---
    valid_csv_files = [
        path for path in all_csv_files
        if derive_target_id_from_path(path) in preprocessed_data
    ]

    if not valid_csv_files:
        print(f"No CSV files in '{input_dir}' correspond to a target_id in '{args.preprocessed_data_path}'.")
        return
        
    print(f"Found {len(valid_csv_files)} files to process that are present in the JSON file.")

    # --- Parallel Execution ---
    futures = []
    filtered_percentages = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        # Use the filtered list 'valid_csv_files' here
        for csv_path in valid_csv_files:
            future = executor.submit(
                process_single_msa,
                csv_path,
                output_dir,
                args.msa_filtering_threshold,
                preprocessed_data,
                args.max_seqs,
            )
            futures.append(future)
        
        # Monitor progress and collect results
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(valid_csv_files), desc="Processing files"):
            result = future.result()
            if result["status"] == "success":
                before = result["before"]
                after = result["after"]
                if before > 0:
                    percent_filtered = ((before - after) / before) * 100
                    filtered_percentages.append(percent_filtered)
            else:
                print(result["message"])# Print any error messages

    print("\nProcessing complete! ✅")
    if filtered_percentages:
        mean_percent = np.mean(filtered_percentages)
        std_dev_percent = np.std(filtered_percentages) 

        print("\n--- Statistics ---")
        print(f"Success number: {len(filtered_percentages)}")
        # 使用 \u00B1 来打印正负号 ±
        print(f"Average filtered number: {mean_percent:.2f} \u00B1 {std_dev_percent:.2f}%")
    else:
        print("\nNo successful processed file.")

if __name__ == "__main__":
    main()
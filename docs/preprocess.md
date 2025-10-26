# Antibody Data Preprocessing Pipeline

This repository contains the data preprocessing pipeline corresponding to Appendix A of our paper. The pipeline processes antibody-antigen complex data from SAbDab and prepares it for training with the Boltz framework.

**Code Location**: All data preprocessing scripts are located in the `scripts/process` directory. You can download raw data and `ccd.rdb` from [https://huggingface.co/datasets/clorf6/MF-Design](https://huggingface.co/datasets/clorf6/MF-Design).

## Overview

The preprocessing pipeline consists of 7 main steps that should be executed in the following order:

1. **Data Summary** (`summary.py`) - Extract basic information from SAbDab raw data
2. **Deduplication** (`decuplication.py`) - Remove entries with highly similar CDR regions
3. **YAML Generation** (`generate.py`) - Generate Boltz-compatible YAML files
4. **Date-based Cutoff** (`cutoff.py`) - Split data by release date to prevent data leakage
5. **CDR Clustering** (`cluster.py`) - Cluster CDR sequences using MMseqs2
6. **Dataset Splitting** (`split.py`) - Create train/validation/test splits without data leakage
7. **NPZ Conversion** (`antibody.py` + `mmcif.py`) - Convert data to Boltz-compatible NPZ format

## Data Processing Notes

**Important Considerations:**

- The entire preprocessing pipeline is based on the Boltz data processing framework. For detailed reference, see: https://github.com/jwohlwend/boltz/blob/main/docs/training.md
- All steps are designed to prevent data leakage and ensure proper train/validation/test separation
- The pipeline specifically handles antibody-antigen complex data with careful attention to CDR region processing
- **Temporal Data Splitting**: Uses CIF file release dates instead of SAbDab-provided dates because the latter are less precise and may not correspond to actual release dates
- **CDR-based Clustering**: Following DiffAb and AbDiffuser methodologies, data splitting is based on CDR-H3 sequence identity to prevent structural similarity leakage
- **MMseqs2 Version**: Uses Release 15-6f452 specifically, which works well for short sequences (CDRs). Newer versions like Release 16-747c6 may have issues with short sequences

## Prerequisites

### Installation

1. **Install MMseqs2 Release 15-6f452**:
   ```bash
   # Download from https://github.com/soedinglab/MMseqs2/releases/tag/15-6f452
   # Follow the installation instructions for your platform
   ```

2. **Setup Redis server** (Only required for step 7):
   ```bash
   redis-server --dbfilename ccd.rdb --port 7777
   ```

## Pipeline Steps

### Step 1: Data Summary (`summary.py`)

Generates summary files from SAbDab raw data containing basic information about antibodies and their corresponding antigens.

**Key Parameters:**
- `--consider_no_antigen`: Include entries without antigens in the summary (optional flag)
- `--chothia_dir`: Directory containing PDB files in Chothia numbering scheme
- `--summary_dir`: Path to SAbDab summary TSV file
- `--output_dir`: Output directory for generated files
- `--output_file_name`: Name of output JSON file (default: 'summary.json')

**Input**: SAbDab raw data files (PDB + summary TSV)
**Output**: 
- `summary.csv` - Tabular summary of antibody-antigen complexes
- `summary.json` - JSON format summary with detailed information
- Log files in `logs-for-preprocess/` directory

**Usage**:
```bash
python summary.py \
    --chothia_dir /path/to/chothia/pdb/files \
    --summary_dir /path/to/sabdab_summary_all.tsv \
    --output_dir /path/to/output \
    --consider_no_antigen
```

### Step 2: Deduplication (`decuplication.py`)

Removes redundant entries by keeping only one representative for antibodies with highly similar CDR regions using Levenshtein distance.

**Key Parameters:**
- `--distance_threshold`: Levenshtein distance threshold for CDR similarity (default: 9, recommended)
- `--original_data_path`: Path to summary CSV from Step 1
- `--output_dir`: Output directory for deduplicated data
- `--output_file_name`: Base name for output file (without extension)

**Input**: Summary CSV from Step 1
**Output**: Deduplicated CSV file with suffix indicating distance threshold

**Usage**:
```bash
python decuplication.py \
    --distance_threshold 9 \
    --original_data_path /path/to/summary.csv \
    --output_dir /path/to/output \
    --output_file_name summary-decuplication
```

**Note**: Distance threshold of 9 is empirically chosen as optimal for antibody CDR deduplication.

### Step 3: YAML Generation (`generate.py`)

Converts deduplicated data into YAML files compatible with the Boltz framework format requirements.

**Key Parameters:**
- `--processed_data_dir_after_decuplication`: Path to deduplicated CSV file from Step 2
- `--output_dir`: Output directory for YAML files

**Input**: Deduplicated CSV from Step 2
**Output**: Individual YAML files for each antibody-antigen complex

**Usage**:
```bash
python generate.py \
    --processed_data_dir_after_decuplication /path/to/summary-decuplication-distance_threshold_9.csv \
    --output_dir /path/to/yaml/output
```

### Step 4: Date-based Cutoff (`cutoff.py`)

Performs temporal data splitting using CIF file release dates to prevent data leakage. Uses precise release dates from CIF files rather than less accurate SAbDab dates.

**Key Parameters:**
- `--summary_file_path`: Path to deduplicated summary CSV
- `--cif_dir`: Directory containing CIF files for date extraction
- `--cutoff_date`: Date cutoff in YYYY-MM-DD format (default: '2021-09-30')
- `--out_dir`: Output directory for cutoff results

**Input**: Deduplicated CSV and CIF files directory
**Output**: JSON file listing PDB IDs before cutoff date

**Usage**:
```bash
python cutoff.py \
    --summary_file_path /path/to/summary-decuplication-distance_threshold_9.csv \
    --cif_dir /path/to/cif/files \
    --cutoff_date 2021-09-30 \
    --out_dir /path/to/output
```

### Step 5: CDR Clustering (`cluster.py`)

Clusters CDR sequences using MMseqs2 for similarity analysis and grouping. Extracts CDR-H3 region from both heavy and light chains.

**Key Parameters:**
- `--processed_csv_path`: Path to processed CSV file
- `--mmseqs_path`: Path to MMseqs2 binary executable
- `--seq_identity_threshold`: Sequence identity threshold for clustering (default: 0.5)
- `--decuplication_threshold`: Should match the threshold used in Step 2 (default: 9)
- `--out_dir`: Output directory for clustering results

**Input**: Processed CSV with CDR sequences
**Output**: 
- FASTA files for each CDR region (heavy_chain_cdr1/2/3.fasta, light_chain_cdr1/2/3.fasta)
- Clustering results in TSV format

**Usage**:
```bash
python cluster.py \
    --processed_csv_path /path/to/summary-decuplication-distance_threshold_9.csv \
    --mmseqs_path /path/to/mmseqs/bin/mmseqs \
    --seq_identity_threshold 0.5 \
    --decuplication_threshold 9 \
    --out_dir /path/to/output
```

**Important**: Requires MMseqs2 Release 15-6f452 for optimal performance with short CDR sequences.

### Step 6: Dataset Splitting (`split.py`)

Creates train/validation/test splits while ensuring no data leakage between sets based on CDR-H3 clustering and temporal constraints.

**Key Parameters:**
- `--clu_result_path`: Path to clustering TSV result (typically heavy_chain_cdr3 cluster)
- `--pdb_before_cutoff_in_sabdab_path`: Path to JSON file from Step 4
- `--precessed_data_path`: Path to processed CSV data
- `--output_dir`: Output directory for split files

**Input**: 
- Clustering results from Step 5
- Date cutoff results from Step 4
- Processed data CSV

**Output**: 
- `train_entry.json` - Training set entry list
- `val_entry.json` - Validation set entry list  
- `test_entry.json` - Test set entry list

**Usage**:
```bash
python split.py \
    --clu_result_path /path/to/cluster_results/heavy_chain_cdr3/cluster_heavy_chain_cdr3_cluster.tsv \
    --pdb_before_cutoff_in_sabdab_path /path/to/before_cutoff_in_sabdab.json \
    --precessed_data_path /path/to/summary-decuplication-distance_threshold_9.csv \
    --output_dir /path/to/output
```

**Splitting Strategy**: 90% train, 5% validation, 5% test with temporal and clustering constraints to prevent leakage.

### Step 7: NPZ Conversion (`antibody.py` + `mmcif.py`)

Converts processed data into Boltz-compatible NPZ format with AF3-style filtering. 

**Important Data Processing Features:**
- **CDR/Epitope Labeling**: For each residue and atom, an `is_cdr` attribute is added to mark whether the residue or atom belongs to CDR regions or antigen epitopes. This attribute will be used during training for targeted noise addition and to distinguish between different regions.
- **Atom Filtering for Design Regions**: For training design regions (all CDR regions by default, or regions specified by `--cdr_select` parameter), only backbone atoms and CB atoms are retained to prevent data leakage caused by varying atom counts across different residue types.

**Files**:
- `antibody.py` - Main conversion script
- `mmcif.py` - Required utility library for structure processing

**Key Parameters:**
- `--datadir`: Directory containing PDB files (default: chothia directory)
- `--processed_csv_fpath`: Path to processed CSV file
- `--outdir`: Output directory for NPZ files
- `--cif_path`: Path to CIF files directory
- `--yaml_path`: Path to YAML files from Step 3
- `--num_processes`: Number of parallel processes (default: CPU count)
- `--redis_host`: Redis server host (default: 'localhost')
- `--redis_port`: Redis server port (default: 7777)
- `--cdr_select`: Specific CDR region to select (optional, e.g., 'H3', 'L3'. By default, all CDRs will be considered)

**Prerequisites**:
```bash
# Start Redis server first
redis-server --dbfilename ccd.rdb --port 7777
```

**Usage**:
```bash
python antibody.py \
    --datadir /path/to/chothia/pdb/files \
    --processed_csv_fpath /path/to/summary-decuplication-distance_threshold_9.csv \
    --outdir /path/to/npz/output \
    --cif_path /path/to/cif/files \
    --yaml_path /path/to/yaml/files \
    --num_processes 8 \
    --redis_host localhost \
    --redis_port 7777
```

**Applied Filters**:
- `ExcludedLigands()` - Removes problematic ligands
- `MinimumLengthFilter(min_len=4, max_len=5000)` - Length constraints
- `UnknownFilter()` - Removes unknown residues
- `ConsecutiveCA(max_dist=10.0)` - Ensures structural continuity
- `ClashingChainsFilter(freq=0.3, dist=1.7)` - Removes clashing structures

**Filtering Criteria**:
- Entries without epitopes are excluded
- Entries that fail AF3 filtering criteria are excluded
- Only valid antibody-antigen complexes with complete structural information are retained

**Output Structure**:
- `structures/` - NPZ files with structural data
- `records/` - JSON files with metadata and records

---

## MSA Data Processing

For Multiple Sequence Alignment (MSA) data, there are two possible workflows:

### Option 1: Using Pre-computed MSA Data

If you already have pre-processed `.csv` files containing MSA data, you can run the following script to perform similarity filtering and generate the final `.npz` files:

```bash
python convert_msa.py \
    --input_dir /path/to/csv/files \
    --output_dir /path/to/npz/output \
    --preprocessed_data_path /path/to/summary.json \
    --msa_filtering_threshold 0.2
```

### Option 2: Generating MSA Data Locally

If you do not have MSA data, you can set up a local MSA server and process the data from scratch. Please refer to the guide at [scripts/process/local_msa/note.md](scripts/process/local_msa/note.md) for detailed instructions.

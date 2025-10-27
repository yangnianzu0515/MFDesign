 # MFDesign: Antibody Sequence and Structure Co-design

[![Paper](https://img.shields.io/badge/Paper-NeurIPS%202025-blue)](https://openreview.net/pdf/fa449e7c0c04e638dab5d8b500e85cbb30f29694.pdf)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Repurposing AlphaFold3-like Protein Folding Models for Antibody Sequence and Structure Co-design**

*Accepted at NeurIPS 2025*

## Overview

MFDesign is a novel approach that repurposes AlphaFold3-style protein folding models for antibody sequence and structure co-design. By adapting state-of-the-art protein structure prediction models, we enable simultaneous optimization of both antibody sequence and structure, providing a powerful tool for antibody engineering and design.

Our method builds upon the open-source [Boltz-1](https://github.com/jwohlwend/boltz) framework, extending its capabilities to handle the unique challenges of antibody design, including CDR region optimization and antigen-antibody interaction modeling.

## Installation

### Prerequisites

MFDesign requires the same environment as Boltz-1. First, install the Boltz framework:

```bash
pip install boltz -U
```

### Setup

1. Clone the MFDesign repository
```bash
git clone https://github.com/yangnianzu0515/MFDesign.git
cd MFDesign
```

2. Download Data and Models:
We provide all processed data, raw source data, and pre-trained models in a separate repository on Hugging Face Hub. You can find them in the ./data and ./model directories within the corresponding repository. The model repository and data repository are available at [MF-Design Model](https://huggingface.co/clorf6/MF-Design) and [MF-Design Data](https://huggingface.co/datasets/clorf6/MF-Design), respectively. After downloading the data, you will also need to unzip the compressed files.

3. Modify the system path in `train.py` and `predict.py` to point to your codebase:
```python
import sys
sys.path.insert(0, '/$YOURPATH/MFDesign/src')
```

## Usage

### Training

For detailed training instructions, see [docs/training.md](docs/training.md).

Basic training command:
```bash
python scripts/train/train.py scripts/train/configs/stage_1.yaml
```

### Inference

For detailed prediction instructions, see [docs/predict.md](docs/predict.md).

Basic prediction command:
```bash
python scripts/predict.py --data <INPUT_PATH> --use_msa_server
```

### Data Preprocessing

We provide both the pre-processed data and the original raw data. For users who wish to run the preprocessing pipeline themselves, please follow the comprehensive instructions in [docs/preprocess.md](docs/preprocess.md).

### Local MSA Generation

For local MSA data processing instructions, see [scripts/process/local_msa/note.md](scripts/process/local_msa/note.md).

## Paper and Citation

If you use MFDesign in your research, please cite our paper:

```bibtex
@article{MFDesign,
  title={Repurposing AlphaFold3-like Protein Folding Models for Antibody Sequence and Structure Co-design},
  author={Nianzu Yang and Songlin Jiang and Jian Ma and Huaijin Wu and Shuangjia Zheng and Wengong Jin and Junchi Yan},
  journal={NeurIPS 2025},
  year={2025}
}
```

## Acknowledgments

This work is built upon the excellent [Boltz-1](https://github.com/jwohlwend/boltz) framework. We thank the Boltz-1 team for their outstanding contributions to the protein structure prediction community and for making their code openly available.

Welcome to contact us via [clorf6@sjtu.edu.cn](mailto:clorf6@sjtu.edu.cn) or [majian7@sjtu.edu.cn](mailto:majian7@sjtu.edu.cn) for any question (the first author Nianzu will not be able to respond to your questions as he is about to start working and will not have much time to continue research and answer questions).

# GADA: A Graph-Based Dual Attention Model for Chest Temporal Disease Tracking and Radiology Report Generation

**GADA** is a novel deep learning framework designed to enhance Automated Radiology Report Generation (ARRG) from chest X-rays. It introduces a **Symptoms-Disease Progression Graph (SPG)** to model temporal clinical knowledge, and a **Graph-based Dual Attention Mechanism (GDAM)** to align evolving disease features with visual regions of interest.



#  Important Notice

 This code is **directly associated with our manuscript submitted to the _The Visual Computer_ journal**.  
If you use this repository in your research, **please cite the corresponding paper** (see citation section below).  
We encourage transparency and reproducibility in medical AI. This repository provides **full implementation**, **setup instructions**, and **evaluation tools** to replicate our results.



#  Key Features

- **Time-aware Clinical Graph (SPG)**: Models disease progression through forward/backward temporal edges.
- **Graph-based Dual Attention (GDAM)**: Integrates visual and clinical cues using DGA (Dynamic Graph Attention) and KEA (Key Event Attention).
- **Hybrid Positional Encoding (HTPE)**: Enhances long-sequence textual decoding.
- **End-to-End Trainable**: Optimized for IU-Xray and MIMIC-CXR datasets.
- **Reproducible & Interpretable**: Designed with clarity, ablation, and modularity in mind.



#  Dependencies and Environment

Ensure you have the following installed:

- Python ≥ 3.7  
- PyTorch ≥ 1.7  
- `transformers`, `numpy`, `matplotlib`, `scikit-learn`, `opencv-python`

You can install dependencies with:

```bash
pip install -r requirements.txt
```
##  Algorithm Modules Overview

### Symptoms-Disease Progression Graph (SPG)

* Time-aware graph with nodes as symptoms/diseases
* Bi-directional edges represent disease transitions over time

###  Graph-based Dual Attention Mechanism (GDAM)

* Combines DGA (graph-enhanced self-attention) + KEA (key event gate)
* Allows focus on evolving regions and symptoms
* Implemented in `modules/model.py`

###  Hybrid Transformer Positional Encoding (HTPE)

* Combines sinusoidal and learned positional vectors
* Improves long report generation
* Implemented in `modules/model.py`

 ##  Dataset

Download the following datasets and place them under the `data/` directory:

* [IU X-Ray Dataset](https://iuhealth.org/find-medical-services/x-rays)
* [MIMIC-CXR Dataset](https://physionet.org/content/mimic-cxr-jpg/2.0.0/)

Expected directory structure:

```
GADA/
├── config/
├── data/
│   ├── iu_xray/
│   └── mimic_cxr/
├── models/
├── modules/
│   ├── dataloader/
│   ├── model/
    ├── ..../
│   ├── loss/
│   ├── metrics/
│   ├── tokenizer/
│   └── utils/
├── results/
├── pycocoevalcap/
├── main_train.py
├── main_test.py
└── README.md
```
## Train

Run `bash train_iu_xray.sh` to train a model on the IU X-Ray data.

Run `bash train_mimic_cxr.sh` to train a model on the MIMIC-CXR data.

## Test

Run `bash test_iu_xray.sh` to test a model on the IU X-Ray data.

Run `bash test_mimic_cxr.sh` to test a model on the MIMIC-CXR data.


# Results
| Model    | BLEU-4    | METEOR    | ROUGE-L   | CIDEr     |
| -------- | --------- | --------- | --------- | --------- |
| **IU-Xray** | **0.191** | **0.207** | **0.401** | **0.371** |
| **MIMIC** | **0.118** | **0.158** | **0.293** | **0.230** |

(Refer to the paper for full comparison)
#  Configuration

Edit configuration files inside the `maintrain/` directory to set:

* Dataset paths
* Training hyperparameters
* Model saving/loading options

# Acknowledgments

This work is supported by a grant from the **Natural Science Foundation of China (Grant No. 62072070)**.  <br><br>

We would also like to express our gratitude to all the source code contributors, especially the authors of **R2Gen**, whose work inspired parts of this implementation.


# Citation
If you find this work helpful, please cite our paper:<br>
```bibtex
@article{usman2025gada,
  title={GADA: Enhancing Radiology Report Generation with Graph-based Dual Attention and Temporal Disease Progression Modeling},
  author={Usman M, et al.},
  journal={The Visual Computer},
  year={2025},
  note={This code is part of the manuscript submitted to *The Visual Computer*.  
If you use this work, please cite our paper.
}


```



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

- Python ≥ 3.8  
- PyTorch ≥ 1.7  
- `transformers`, `numpy`, `matplotlib`, `scikit-learn`, `opencv-python`

You can install dependencies with:

```bash
pip install -r requirements.txt
```
##  Algorithm Modules Overview

### **Algorithm Documentation**  

#### **1. FEP**    Extracts visual features from chest X-rays using ResNet-101 
#### **2. Symptoms-Disease Progression Graph (SPG)**  
- **Function**: Graph structure capturing temporal disease progression and co-occurrence 
- **Structure**:  
  - **Nodes**: Symptoms/diseases  
  - **Edges**: Bidirectional (disease transitions across imaging studies)  
  

#### **3. Graph-based Dual Attention Mechanism (GDAM)**  
- **Components**:  
  - **DGA (Dynamic Graph Attention)**: Refines SPG embeddings by dynamically updating them based on time-aware dependencies 
  - **KEA (Key Event Attention)**: GaAttends to key clinical events using adaptive gating to highlight important findings  
- **Purpose**: Focuses on anatomically evolving regions  
- **Implementation**: `modules/model.py`  

#### **4. Hybrid Transformer Positional Encoding (HTPE)**  
- **Design**:  
  - Sinusoidal (fixed) + Learned (adaptive) positional vectors  
- **Advantage**: Captures long-range dependencies and handles long diagnostic reports and contextual bias
- **Implementation**: `modules/model.py`  
- **Transformer Decoder**  Generates clinically accurate and coherent radiology reports 

## Pseudocode
```python
# Input: Sequential Chest X-ray images I = {I1, I2, ..., In}
# Output: Radiology Report Y

# Step 1: Feature Extraction (FEP)
for image in I:
    visual_features = ResNet101(image)   # Extract deep features
    I_features.append(visual_features)

# Step 2: Build Symptom-Disease Progression Graph (SPG)
SPG = create_graph(nodes=symptoms, edges=temporal_relationships)
SPG_embeddings = RGCN(SPG)               # Learn initial graph embeddings

# Step 3: Apply Graph-Based Dual Attention Mechanism (GDAM)

# Dynamic Graph Attention (DGA) to refine graph node embeddings
V_prime = SPG_embeddings
V_double_prime = DGA(V_prime, visual_features, SPG.adjacency_matrix)

# Key Event Attention (KEA) to focus on critical findings over time
attention_output = KEA(visual_features, V_double_prime)

# Fuse graph and visual information
G_fused = FullyConnectedLayer(attention_output)

# Step 4: Positional Encoding with HTPE
G_encoded = HTPE(G_fused, sequence_length=L)

# Step 5: Report Generation using Transformer Decoder
Q, K, V = LinearProjection(G_encoded)
encoder_output = TransformerEncoder(Q, K, V)

# Decode the report sequence token by token
Y = TransformerDecoder(encoder_output)

# Step 6: Training Objective
# Maximize conditional probability of generating report Y given input images and SPG
loss = CrossEntropy(Y_pred, Y_true)
optimize(loss)

# Final Output: Generated Report Y
return Y
```

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


## Citation 
```
If you use this code or findings, please cite:  

@article{usman2025gada,  
  title = {GADA: Enhancing Radiology Report Generation with Graph-Based Dual Attention and Temporal Disease Progression Modeling},  
  author = {Usman, M. and [Coauthors]},  
  journal = {The Visual Computer},  
  year = {2025},  
  doi = {10.5281/zenodo.15771095},  
  note = {Code: \url{https://github.com/Usmannooh/GADA}}  
}  

*This repository accompanies the manuscript under review at [The Visual Computer].*  

```



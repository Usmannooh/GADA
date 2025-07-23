# GADA: Enhancing Radiology Report Generation with Graph-Aware Dual Attention and Temporal Disease Progression Modeling

This paper focuses on the task of automated radiology report generation (ARRG) from chest X-ray images. Accurate and interpretable ARRG is essential for enhancing diagnostic efficiency, reducing radiologist workload, and improving patient outcomes in clinical workflows. While recent methods leverage vision-language models and medical knowledge graphs, they predominantly rely on static relationships and lack the ability to model evolving disease progression over time, a critical factor in real-world radiological assessment. These limitations raise two fundamental challenges:
(1) How can we effectively model temporal disease evolution across sequential imaging studies to improve the contextual accuracy of generated reports?  (2) How can we integrate structured clinical knowledge early in the generation pipeline to enhance interpretability and reduce reporting bias?  To address these challenges, we propose GADA, a novel graph-based dual-attention framework for radiology report generation. 

**GADA** is a novel deep learning framework designed to enhance Automated Radiology Report Generation (ARRG) from chest X-rays. It introduces a Symptoms-Disease Progression Graph (SPG) to model temporal clinical knowledge, and a Graph-based Dual Attention Mechanism (GDAM) to align evolving disease features with visual regions of interest.



#  Important Notice

 This code is **directly associated with our manuscript submitted to the _Computer Methods and Programs in Biomedicine**.  
If you use this repository in your research, **please cite the corresponding paper** (see citation section below).  
We encourage transparency and reproducibility in medical AI. This repository provides **full implementation**, **setup instructions**, and **evaluation tools** to replicate our results.



# Key Features

- **Symptom–Disease Progression Graph (SPG-MKG):** Captures temporal disease dynamics using forward/backward edges in a clinically grounded knowledge graph.
- **Bidirectional Graph Reasoning (BiRGR):** Learns contextual representations over symptom–disease relationships using stacked graph convolutions.
- **Class-Specific Spatial Aggregation (CSSA):** Enhances visual localization by aggregating CNN features per disease class.
- **Context-Aware Dual Attention (CADA + TSA):** Aligns visual and clinical cues via semantic-aware (CADA) and saliency-driven (TSA) mechanisms.
- **Context-Guided Temporal Positional Encoding (CTPE):** Maintains temporal consistency across multi-visit image sequences.
- **End-to-End Trainable & Interpretable:** Modular, reproducible design validated on IU-Xray and MIMIC-CXR with strong clinical coherence and ablation support.




#  Dependencies and Environment

Ensure you have the following installed:

- Python ≥ 3.8  
- PyTorch ≥ 1.7  
- `transformers`, `numpy`, `matplotlib`, `scikit-learn`, `opencv-python`, `tensorflow-gpu`,  `pandas`, `cudatoolkit=11.7`, `nltk`

You can install dependencies with:

```bash
pip install -r requirements.txt
```
##  Algorithm Modules Overview

 **1. Feature Extraction Process (FEP)**
- **Function**: Extracts spatial visual features from multi-view chest X-rays using a pre-trained ResNet-101.
- **Enhancement**: Applies Class-Specific Spatial Aggregation (CSSA) to focus on disease-relevant regions before fusion.


 **2. Symptom–Disease Progression Graph (SPG-MKG)**
- **Function**: Encodes prior medical knowledge and captures temporal disease evolution across patient visits.
- **Structure**:
  - **Nodes**: Thoracic symptoms and diseases (e.g., opacity, effusion)
  - **Edges**: Bidirectional temporal relations (forward and backward adjacency)
- **Personalization**: Activated dynamically using visual-symptom alignment and inter-visit similarity.



### **3. Graph Reasoning and Dual Attention Module**
- **Components**:
  - **BiRGR (Bidirectional Relational Graph Reasoner)**: Performs context-aware reasoning over SPG using stacked GCN layers.
  - **CADA (Context-Aware Dynamic Alignment)**: Aligns symptom-visual representations based on semantic cues.
  - **TSA (Temporal Saliency Attention)**: Highlights key clinical events across temporal visit sequences.
- **Purpose**: Models evolving clinical semantics and spatial-temporal alignment.
- **Implementation**: `modules/attention.py`, `modules/graph.py`



### **4. Context-Guided Temporal Positional Encoding (CTPE)**
- **Design**: Combines sinusoidal encodings with visit-aware temporal masking (`pt`) and report-aware contextual cues (`rt`).
- **Function**: Injects chronological coherence into Transformer input representations.
- **Implementation**: `modules/position.py`



### **5. Report Generation via Hybrid Transformer**
- **Design**: Multi-layer Transformer encoder-decoder conditioned on fused clinical and visual embeddings.
- **Function**: Generates coherent, interpretable radiology reports aligned with disease evolution.
- **Implementation**: `modules/transformer.py`



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
  journal = {},  
  year = {2025},  
  doi = {Submitted},  
  note = {Code: \url{https://github.com/Usmannooh/GADA}}  
}  

*This repository accompanies the manuscript under review .*  

```



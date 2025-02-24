# GADA: A Graph-Based Dual Attention Model for Chest Temporal Disease Tracking and Radiology Report Generation

 A Graph-Based Dual Attention Model for Chest Temporal Disease Tracking and Radiology Report Generation (GADA) is a novel framework that addresses key challenges in Automated Radiology Report Generation (ARRG). It integrates **Symptoms-disease progression graph (SPG)** for temporal reasoning and advanced and **Dual-based attention mechanisms** to generate accurate, interpretable, and contextually relevant diagnostic reports from medical images. 


## Introduction

Automated Radiology Report Generation (ARRG) leverages machine learning to interpret medical images and generate structured diagnostic reports. The **GADA** framework addresses three core challenges in ARRG:

1. **Modeling Temporal Disease Progression**: ISRA dynamically captures **spatial** and **temporal relationships** in clinical data .
2. **Multimodal Fusion**: Integrates **visual features** from medical images with **clinical context** from textual data using **Relational Graph Convolutional Networks (RGCNs)**.
3. **Interpretability**: Enhances the explainability of the generated reports using advanced **DAM**, allowing clinicians to better trust and validate the AI's output.

## Features

- **Temporal Reasoning **: Models the progression of diseases over time based on serial imaging and clinical observations.
- **Multimodal Fusion**: Combines visual and textual features for more comprehensive diagnostic reports.
- **Advanced Attention Mechanisms**: Uses **Dynamic Graph Attention (DGA)** and **Key Event Attention (KEA)** to integrate clinical knowledge with visual patterns.
- **State-of-the-Art Performance**: Outperforms existing ARRG models with significant improvements in BLEU, METEOR, and ROUGE-L metrics.
  
## Installation

To get started with ISRA, follow the steps below:

## Prerequisites

- Python 3.7 or later
- Dependencies: `torch`, `numpy`, `matplotlib`, `scikit-learn`, `transformers`

## Acknowledgments

This work is supported by a grant from the **Natural Science Foundation of China (Grant No. 62072070)**.  <br><br>

We would also like to express our gratitude to all the source code contributors, especially the authors of **R2GenCMN**, whose work inspired parts of this implementation.


## Citation
If you find this work helpful, please cite our paper:<br>
```bibtex


```



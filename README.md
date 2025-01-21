# ISRA - Integrating Symptoms Trend for Radiology Report Generation

**ISRA** (Integrating Symptoms Trend for Radiology Report Generation) is a novel framework designed to address key challenges in Automated Radiology Report Generation (ARRG). It integrates **Symptoms Trend Knowledge Integration (STD-KI)** for temporal reasoning, advanced **multimodal fusion**, and **attention mechanisms** to generate accurate, interpretable, and contextually relevant diagnostic reports from medical images. 


## Introduction

Automated Radiology Report Generation (ARRG) leverages machine learning to interpret medical images and generate structured diagnostic reports. The **ISRA** framework addresses three core challenges in ARRG:

1. **Modeling Temporal Disease Progression**: ISRA dynamically captures **spatial** and **temporal relationships** in clinical data using **Symptoms Trend Knowledge Integration (STD-KI)**.
2. **Multimodal Fusion**: Integrates **visual features** from medical images with **clinical context** from textual data using **Relational Graph Convolutional Networks (RGCNs)**.
3. **Interpretability**: Enhances the explainability of the generated reports using advanced **attention mechanisms**, allowing clinicians to better trust and validate the AI's output.

## Features

- **Temporal Reasoning with STD-KI**: Models the progression of diseases over time based on serial imaging and clinical observations.
- **Multimodal Fusion**: Combines visual and textual features for more comprehensive diagnostic reports.
- **Advanced Attention Mechanisms**: Uses **Dynamic Graph Attention (DGA)** and **Key Event Attention (KEA)** to integrate clinical knowledge with visual patterns.
- **State-of-the-Art Performance**: Outperforms existing ARRG models with significant improvements in BLEU, METEOR, and ROUGE-L metrics.
  
## Installation

To get started with ISRA, follow the steps below:

### Prerequisites

- Python 3.7 or later
- Dependencies: `torch`, `numpy`, `matplotlib`, `scikit-learn`, `transformers`



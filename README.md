# GADA: A Graph-Based Dual Attention Model for Chest Temporal Disease Tracking and Radiology Report Generation

**GADA** is a novel deep learning framework designed to enhance Automated Radiology Report Generation (ARRG) from chest X-rays. It introduces a **Symptoms-Disease Progression Graph (SPG)** to model temporal clinical knowledge, and a **Graph-based Dual Attention Mechanism (GDAM)** to align evolving disease features with visual regions of interest.

---

#  Important Notice

 This code is **directly associated with our manuscript submitted to the _The Visual Computer_ journal**.  
If you use this repository in your research, **please cite the corresponding paper** (see citation section below).  
We encourage transparency and reproducibility in medical AI. This repository provides **full implementation**, **setup instructions**, and **evaluation tools** to replicate our results.

---

#  Key Features

- **Time-aware Clinical Graph (SPG)**: Models disease progression through forward/backward temporal edges.
- **Graph-based Dual Attention (GDAM)**: Integrates visual and clinical cues using DGA (Dynamic Graph Attention) and KEA (Key Event Attention).
- **Hybrid Positional Encoding (HTPE)**: Enhances long-sequence textual decoding.
- **End-to-End Trainable**: Optimized for IU-Xray and MIMIC-CXR datasets.
- **Reproducible & Interpretable**: Designed with clarity, ablation, and modularity in mind.

---

#  Dependencies and Environment

Ensure you have the following installed:

- Python ≥ 3.7  
- PyTorch ≥ 1.7  
- `transformers`, `numpy`, `matplotlib`, `scikit-learn`, `opencv-python`

You can install dependencies with:

```bash
pip install -r requirements.txt




# Acknowledgments

This work is supported by a grant from the **Natural Science Foundation of China (Grant No. 62072070)**.  <br><br>

We would also like to express our gratitude to all the source code contributors, especially the authors of **R2GenCMN**, whose work inspired parts of this implementation.


# Citation
If you find this work helpful, please cite our paper:<br>
```bibtex


```



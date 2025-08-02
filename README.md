# 🎥 UVOS: Unsupervised Video Object Segmentation

A research-focused framework for **Unsupervised Video Object Segmentation (UVOS)** using **contrastive learning** on benchmark datasets including **DAVIS**, **FBMS**, and **YT-OBJ**.

## 🧠 Key Idea

We aim to segment primary foreground objects in videos without manual annotations, leveraging:
- Temporal consistency
- Appearance contrastive features
- Flow-guided supervision
- No ground-truth masks required

## 🛠️ Tech Stack

- Python, PyTorch
- RAFT for Optical Flow
- DAVIS 2016/2017, FBMS, YT-OBJ datasets

## 📂 Datasets Used

- [DAVIS 2016](https://davischallenge.org/davis2016/code.html)
- [FBMS](https://github.com/tfzhou/ASE-Fast)
- [YouTube-Objects](https://github.com/liulu112601/MBNM)

## 🔧 Setup

1. Download the datasets: DUTS, DAVIS, FBMS, YouTube-Objects, Long-Videos.

2. Estimate and save optical flow maps from the videos using RAFT.

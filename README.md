# 🎥 UVOS: Unsupervised Video Object Segmentation

A research-focused framework for **Unsupervised Video Object Segmentation (UVOS)** using **Contrastive learning** on benchmark datasets including **DAVIS**, **FBMS**, and **YT-OBJ**.

### Demo

![BiCycle Demo](BiCycle.gif)


## 🧠 Key Idea

We aim to segment primary foreground objects in videos without manual annotations, leveraging:
- Temporal consistency
- Appearance contrastive features
- Flow-guided supervision
- No ground-truth masks required

## 🛠️ Tech Stack

- Python, PyTorch
- RAFT for Optical Flow

## 📂 Datasets Used

- [DAVIS 2016](https://davischallenge.org/davis2016/code.html)
- [FBMS](https://github.com/tfzhou/ASE-Fast)
- [DUTS](https://dut-omron.github.io/DUTS)
- [YouTube-Objects](https://github.com/liulu112601/MBNM)

## 🔧 Setup

1. Download the datasets: DUTS, DAVIS, FBMS, YouTube-Objects.

2. Estimate and save optical flow maps from the videos using RAFT.


🚀 Running

🏋️‍♂️ Training
To start VISE training, run:


    python run.py --train
    
Verify the following before running:
✅ Training dataset selection and configuration
✅ GPU availability and configuration
✅ Backbone network selection


🧪 Testing

      python run.py --test

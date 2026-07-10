# Satellite Image Change Detection for Disaster Response

> Final Year B.Tech Capstone Project | Computer Science & Engineering (AI & ML)

## 📌 Overview

This project focuses on developing an intelligent deep learning framework for automatic disaster damage assessment using multi-temporal satellite imagery. By comparing pre-disaster and post-disaster satellite images, the proposed system aims to accurately identify structural changes and damaged regions, enabling faster and more efficient disaster response.

The project begins with binary change detection using the **LEVIR-CD** dataset and later extends to multi-class disaster damage assessment using the **xBD (xView2)** dataset. Advanced deep learning architectures, including transformer-based change detection models, will be investigated and evaluated.

---

## 🎯 Objectives

- Develop a robust deep learning model for satellite image change detection.
- Detect structural changes between pre-disaster and post-disaster imagery.
- Address class imbalance using specialized loss functions.
- Compare CNN-based and Transformer-based architectures.
- Evaluate models using standard segmentation metrics.
- Build a reproducible research pipeline suitable for publication.

---

## 🧠 Research Problem

Manual assessment of disaster damage from satellite imagery is time-consuming and resource-intensive. This project aims to automate the process using computer vision and deep learning techniques, assisting emergency response teams in rapidly identifying affected regions.

---

## 📂 Datasets

### Phase 1
- **LEVIR-CD**
  - Binary building change detection
  - High-resolution satellite imagery
  - Pixel-wise binary masks

### Phase 2
- **xBD (xView2)**
  - Multi-disaster dataset
  - Multi-class building damage assessment
  - Earthquakes, floods, hurricanes, wildfires, and more

---

## 🛠️ Technology Stack

### Programming Language
- Python

### Deep Learning
- PyTorch
- TorchVision

### Data Processing
- NumPy
- Pandas
- OpenCV
- Pillow
- Albumentations

### Visualization
- Matplotlib

### Development
- Jupyter Notebook
- Git
- GitHub

---

## 📁 Project Structure

```text
capstone/
│
├── data/
│   ├── LEVIR-CD/
│   └── xBD/
│
├── notebooks/
│   ├── 01_Dataset_Analysis.ipynb
│   ├── 02_Data_Preprocessing.ipynb
│   ├── 03_Baseline_Model.ipynb
│   ├── 04_Model_Training.ipynb
│   ├── 05_Evaluation.ipynb
│   └── ...
│
├── src/
│   ├── datasets/
│   ├── models/
│   ├── trainer/
│   ├── losses/
│   ├── metrics/
│   ├── utils/
│   ├── inference/
│   └── visualization/
│
├── checkpoints/
├── outputs/
├── papers/
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🚀 Research Workflow

- Dataset Analysis
- Data Preprocessing
- Data Augmentation
- Baseline Model Development
- Transformer-Based Model Development
- Model Training
- Evaluation and Benchmarking
- Experimental Analysis
- Research Paper Preparation

---

## 📊 Evaluation Metrics

The following metrics will be used to evaluate model performance:

- Intersection over Union (IoU)
- Dice Coefficient
- Precision
- Recall
- F1 Score
- Pixel Accuracy

---

## 📈 Current Progress

- [x] Project initialization
- [x] Dataset organization
- [x] LEVIR-CD exploratory data analysis
- [ ] Data preprocessing pipeline
- [ ] Custom PyTorch Dataset
- [ ] Baseline Siamese U-Net
- [ ] Transformer-based model
- [ ] Model evaluation
- [ ] Research paper
- [ ] Final project demonstration

---

## 📚 References

- LEVIR-CD Dataset
- xBD (xView2) Dataset
- ChangeFormer
- BIT (Building Change Detection Transformer)
- Swin Transformer
- SatMAE

---

## 👨‍💻 Authors

**Bala Srivatsa Panigrahi**

B.Tech Computer Science & Engineering (AI & ML)

VIT-AP University

---

## 📄 License

This project is being developed as part of an academic research project. The source code will be made publicly available after the completion of the capstone project and research publication.
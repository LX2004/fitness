---
# 🧱 Framework Overview
![image](https://github.com/user-attachments/assets/db28a092-7d87-4346-bc50-97c144666773)
CRISFitFormer is a transformer-based deep learning framework designed to predict bacterial cell fitness based on CRISPRi guide RNA sequences. The system supports multiple species and integrates biological features for enhanced performance.
---

# 📈 Core Function: Fitness Prediction
This project provides a modular pipeline to:

- Predict cell fitness values using a Transformer model
- Incorporate biological and sequence features
- Support cross-validation training
- Compare across species (e.g., *E. coli*, *Cyanobacteria*, *Staphylococcus aureus*, etc.)

---
## 🧰 Environment Setup

```bash
cd fitness

# Create a conda environment
conda create --name fitness

# Activate the environment
conda activate fitness

# Install dependencies
conda install --file request.txt


# To train a model for predicting E. coli fitness values using a dataset, you may perform the following steps.
run :
```
cd E_coli

cd code

python prediction_transformer_ori_dim_bio_kfold.py

```
# To train a model for predicting Cyanobacteria fitness values using a dataset, you may perform the following steps.
run : 
```
cd Cyanobacteria

cd code

python prediction_transformer_ori_dim_bio_kfold.py

```
# To train a model for predicting staphylococcus fitness values using a dataset, you may perform the following steps.
run : 
```
cd staphylococcus

cd code

python prediction.py

```
# To train a model for predicting E_limosum fitness values using a dataset, you may perform the following steps.
run : 
```
cd E_limosum

cd code

python prediction.py

```
# To train a model for predicting bacillus fitness values using a dataset, you may perform the following steps.
run : 
```
cd bacillus

cd code

python prediction.py

```
#  🧬 CRISFitFormer---Online Platform
**CRISFitFormer** is a deep learning framework and web-based platform for predicting bacterial cell fitness from genome-wide CRISPRi knockdown screens. It integrates both experimental fitness computation and transformer-based predictive modeling to support large-scale functional genomics analysis.
![image](https://github.com/user-attachments/assets/83aa48d5-7c9f-4377-b64e-04a5c1de227a)

## 🌐 Online Platform

👉 Visit the platform: [https://crisfitformer.bioinformatics-syn.org/](https://crisfitformer.bioinformatics-syn.org/)

## 🚀 Key Features

| Module | Description |
|--------|-------------|
| 📊 **Fitness Calculation** | Upload CRISPRi screen data and compute fitness scores directly |
| 🤖 **Fitness Prediction** | Use transformer-based models to predict fitness from guide RNA sequences |
| 🧬 **Essentiality Profiling** | Automatically infer gene essentiality from fitness values |
| 📁 **Custom Input Support** | Accepts user-uploaded guide RNA and growth data |

## 🧪 System Architecture

CRISFitFormer supports two complementary workflows:

### 1. Experimental Fitness Calculation

- Input: Cell growth data from CRISPRi screens
- Process: Fitness calculation → Gene essentiality inference
- Output: CSV/Plot of gene-level essentiality

### 2. Deep Learning-Based Prediction

- Input: sgRNA sequences
- Process: Transformer-based inference → Fitness score → Bad seed detection
- Output: Predicted fitness values, sequence toxicity report


# Result.
The model obtained from training is stored in the "models" folder, and details such as training parameters and metrics are recorded in the corresponding text files.

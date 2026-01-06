---
# 🧱 Framework Overview
<img width="1952" height="1736" alt="image" src="https://github.com/user-attachments/assets/279718cd-cae4-44b7-b5a0-ff78642d4f2b" />

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
```
---

## 🧪 Species-Specific Model Training
### To train a model for predicting E. coli fitness values using a dataset, you may perform the following steps.

```
cd E_coli

cd code

python prediction_transformer_ori_dim_bio_kfold.py

```
### To train a model for predicting Cyanobacteria fitness values using a dataset, you may perform the following steps.

```
cd Cyanobacteria

cd code

python prediction_transformer_ori_dim_bio_kfold.py

```
### To train a model for predicting staphylococcus fitness values using a dataset, you may perform the following steps.

```
cd staphylococcus

cd code

python prediction.py

```
### To train a model for predicting E_limosum fitness values using a dataset, you may perform the following steps.

```
cd E_limosum

cd code

python prediction.py

```
### To train a model for predicting bacillus fitness values using a dataset, you may perform the following steps.

```
cd bacillus

cd code

python prediction.py

```

### To train a model for predicting gRNA efficiency, you may perform the following steps.

```
cd prediction_CRISPRi_efficiency_e_coli

cd code

python prediction_transformer_ori_dim_kfold.py

```

### DNA–gRNA Binding Energy Calculation
The script `energy.py` is used to calculate the **direct binding energy** between a gRNA and its corresponding target DNA sequence.  
It applies a nearest-neighbor thermodynamic model and outputs the calculated binding energies in **kcal/mol**.

To use the script:

1. Open `energy.py`.
2. In the **main function**, modify the list of **gRNA sequences** (`grna_list`) according to your needs.
3. Run the script:

```bash
python energy.py
```
4. The program will automatically compute and print the binding energies for each gRNA, and save a publication-quality bar chart in the current directory.


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
| ✨ **gRNA Optimization** | Generate candidate gRNAs from target sequences (NGG PAM) and rank them using off-target risk, bad-seed screening, binding energy, and predicted efficiency |
| 📁 **Custom Input Support** | Accepts user-uploaded guide RNA and growth data |

## 🧪 System Architecture

CRISFitFormer supports three complementary workflows:

### 1. Experimental Fitness Calculation

- Input: Cell growth data from CRISPRi screens
- Process: Fitness calculation → Gene essentiality inference
- Output: CSV/Plot of gene-level essentiality

### 2. Deep Learning-Based Prediction

- Input: sgRNA sequences
- Process: Transformer-based inference → Fitness score
- Output: Predicted fitness values, sequence toxicity report

### 3. gRNA Design and Optimization

- Input: Target DNA sequence
- Process:
  - PAM scanning (NGG) → candidate gRNA generation
  - Multi-criteria scoring and filtering:
    - Off-target assessment
    - Bad-seed screening
    - Binding energy estimation
    - Efficiency / fitness prediction
  - Ranking → select best gRNA(s)
- Output: Ranked gRNA list (best gRNA), with per-guide reports for efficiency and risk




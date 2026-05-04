# Cervical Cancer Risk Factor Analysis
## 📋 Overview

This project aims to analyze risk factors associated with cervical cancer and build machine learning models to predict the likelihood of developing the disease. 
The analysis utilizes the Kaggle Cervical Cancer Risk Factors dataset and implements a complete MLOps pipeline with experiment tracking using Weights & Biases.

## 🎯 Objectives

- Perform comprehensive data cleaning and preprocessing
- Conduct feature selection to identify key risk factors
- Build and train machine learning models (MLP, Random Forest, etc.)
- Track experiments and model performance using Weights & Biases
- Create reproducible pipelines for data processing and model training

## 📊 Dataset

The dataset contains risk factors for cervical cancer including:
- Demographic information (age, number of pregnancies)
- Behavioral factors (smoking habits, number of sexual partners)
- Medical history (STDs, hormonal contraceptives)
- Examination results (citology, schiller, biopsy)
- Diagnosis outcomes

**Link:** [Kaggle Cervical Cancer Risk Factors Dataset](https://www.kaggle.com/datasets/lovishbansal/cervical-cancer-risk-factor)

# Step 1 - Data Cleaning
- Handling Missing Values
- Removed Missing Values
- Removed Duplicates
- Outlier Detection

# Step 2 - Feature Engineering
- One-Hot Encoding
- RobustScaler
- Feature Selection (RF, MI, Lasso)

# Step 3 - Train/Test Split
- 80% Training data
- 20% Testing data
- Stratified split by target column
- Distribution comparison (KS or Chi2)

# Step 4 - Model Architecture (MLP)
- Input layer: number of selected features
- Hidden layers: 64 → 32 neurons
- Dropout: 0.2 (regularization)
- Output layer: 1 neuron (binary classification)
- Activation: ReLU (hidden), Sigmoid (output)

# Step 5 - Training Configuration
- Loss function: Binary Cross-Entropy (BCEWithLogitsLoss)
- Optimizer: Adam
- Learning rate: 0.001
- Batch size: 32
- Epochs: 100
- Early stopping patience: 10

# Step 6 - Evaluation Metrics
- Accuracy
- F1 Score
- ROC-AUC
- Precision
- Recall
- Loss

# Results
<img width="179" height="224" alt="{AF67D620-FA63-4950-8E61-27FA6A11A5FC}" src="https://github.com/user-attachments/assets/9b0210b4-06cf-4707-b856-8c3473379829" />

# How Reproduce the Results

### Step 1: Clone repository
```
git clone https://github.com/GabrielH20/Projeto-Kaggle-IA.git
cd Projeto-Kaggle-IA
```
# Step 2: Make and Active Venv
```
python -m venv venv
venv\Scripts\activate
```
# Step 3: Requirements
```
pip install -r requirements.txt
```
# Step 4: Wandb Login 
```
wandb login
```
Put in your terminal and will asking for your API_KEY
# Step 5 - Runing Pipelines
```
python main.py
```

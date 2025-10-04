# CancerTypeNet: Tumor Type Classification from Mutational Signatures

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9%2B-red)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A comprehensive machine learning pipeline for predicting cancer types from mutational signatures using multiple classification models.

## 📋 Project Overview

This project implements and compares various machine learning approaches to classify tumor types based on Single Base Substitution (SBS) mutational signatures from the **PCAWG dataset**. We achieve up to **79.5% accuracy** across 22 cancer classes using ensemble methods combining catalog and activity data.

### Key Features

- **Multi-model comparison**: KNN, SVM, Random Forest, Gradient Boosting, DNN, CNN
- **Multi-modal learning**: Combining mutational catalogs (96 channels) and signature activities (65 SBS signatures)
- **Comprehensive evaluation**: Confusion matrices, precision-recall analysis, feature importance
- **Explainable AI**: SHAP values and attention visualization for model interpretability

## 🎯 Results Summary

| Model | Dataset | Best Accuracy |
|-------|---------|---------------|
| **DNN** | WGS PCAWG (catalog + activity) | **79.46%** |
| CNN | WGS PCAWG (catalog + activity) | 78.06% |
| KNN | WGS PCAWG (catalog) | 77.84% |
| Gradient Boosting | WGS PCAWG (combined) | 70% |
| Random Forest | WGS PCAWG (catalog) | 68% |
| SVM | WGS PCAWG (catalog) | 41% |

*Note: Results are based on 80-20 train-test split with stratification*

## 📊 Dataset

### Data Source
Data from [Alexandrov et al. (2020) Nature](https://www.nature.com/articles/s41586-020-1943-3) - "The repertoire of mutational signatures in human cancer"

### Data Types
1. **Mutational Catalogs** (WGS/WES)
   - 96 trinucleotide mutation channels
   - PCAWG and TCGA datasets
   
2. **Signature Activities**
   - 65 SBS mutational signatures
   - Derived from NMF decomposition

### Cancer Types
- **WGS PCAWG**: 37 cancer types (2,780 samples)
- **WES TCGA**: Multiple cancer types with varying sample sizes
- After filtering (≥5 samples): 22 major cancer classes

## 🏗️ Project Structure

```
CancerTypeNet/
├── README.md
├── requirements.txt
├── notebooks/
│   ├── 01_CancerTypeNet_Main.ipynb          # Main analysis pipeline
│   └── 02_Data_Exploration.ipynb            # EDA and visualization
├── project_data/
│   ├── catalogs/
│   │   ├── WGS/
│   │   │   └── WGS_PCAWG.96.csv
│   │   └── WES/
│   │       └── WES_TCGA.96.csv
│   └── activities/
│       ├── WGS/
│       │   └── WGS_PCAWG.activities.csv
│       └── WES/
│           └── WES_TCGA.activities.csv
└── results/
    ├── figures/
    └── models/
```

## 🚀 Getting Started

### Prerequisites

```bash
Python 3.8+
PyTorch 1.9+
scikit-learn
pandas
numpy
matplotlib
seaborn
```

### Installation

```bash
# Clone the repository
git clone https://github.com/XiaoyuYuan19/CancerTypeNet.git
cd CancerTypeNet

# Install dependencies
pip install -r requirements.txt
```

### Quick Start

```python
# Load and preprocess data
from data_preprocessing import load_pcawg_data

catalogs, activities, cancer_types = load_pcawg_data()

# Train a model
from models import train_dnn

model, accuracy = train_dnn(
    X=catalogs, 
    y=cancer_types,
    epochs=2000,
    batch_size=32,
    learning_rate=0.001
)

print(f"Test Accuracy: {accuracy:.2%}")
```

## 🔬 Methodology

### 1. Data Preprocessing
- **Normalization**: StandardScaler for deep learning models, MedianMAD for KNN
- **Class balancing**: Removed cancer types with <5 samples
- **Feature engineering**: Combined 96 mutation channels with 65 signature activities (161 features)

### 2. Model Architectures

#### Deep Neural Network (Best Performance)
```
Input (161) → Dense(128, ReLU, L2, Dropout=0.5) 
           → Dense(64, ReLU, L2, Dropout=0.5) 
           → Dense(22, Softmax)
```
- Optimizer: Adam (lr=0.001)
- Loss: Sparse Categorical Crossentropy
- Training: 2000 epochs, batch_size=32

#### Convolutional Neural Network
```
Conv1D(1→14→18→1) + Dropout 
→ Flatten 
→ Dense(160→300→300→220→37)
```
- Regularization: Dropout (0.05-0.07)
- Optimizer: Adam (lr=0.00006)

### 3. Evaluation Metrics
- **Accuracy**: Overall classification accuracy
- **Confusion Matrix**: Per-class performance
- **F1-Score & Recall**: Handling class imbalance
- **Average Precision**: Better metric for imbalanced datasets

## 📈 Key Findings

1. **Multi-modal fusion improves accuracy**
   - Catalog only: 77.8%
   - Activity only: 71.7%
   - Combined: **79.5%**

2. **C>T mutations are most informative**
   - Contribute maximally to cancer type prediction
   - ATA trinucleotide context is most important

3. **Deep learning outperforms traditional ML**
   - DNN and CNN achieve 5-10% higher accuracy than ensemble methods

4. **Class imbalance significantly affects performance**
   - Larger cancer types (Liver-HCC, Kidney-RCC) achieve >80% accuracy
   - Rare cancer types show 0% accuracy without proper handling

## 🎓 Course Information

**Course**: Machine Learning in Molecular Biosciences (LSI31003)  
**Institution**: University of Helsinki  
**Date**: March 2024  
**Team Project**: Group collaboration

## 📝 Citation

If you use this work, please cite:

```bibtex
@misc{yuan2024cancertypenet,
  title={CancerTypeNet: Tumor Type Classification from Mutational Signatures},
  author={Yuan, Xiaoyu and Team Members},
  year={2024},
  institution={University of Helsinki}
}
```

### Reference Paper
```bibtex
@article{alexandrov2020repertoire,
  title={The repertoire of mutational signatures in human cancer},
  author={Alexandrov, Ludmil B and others},
  journal={Nature},
  volume={578},
  number={7793},
  pages={94--101},
  year={2020},
  publisher={Nature Publishing Group}
}
```

## 🔗 Related Resources

- [COSMIC Mutational Signatures](https://cancer.sanger.ac.uk/cosmic/signatures)
- [PCAWG Project](https://dcc.icgc.org/pcawg)
- [Original Paper](https://www.nature.com/articles/s41586-020-1943-3)

## 📧 Contact

**Xiaoyu Yuan**  
📧 xiaoyuyuan19@gmail.com  
🌐 [Portfolio](https://xiaoyuyuan19.github.io/portfolio/)  
💼 [LinkedIn](http://www.linkedin.com/in/xiaoyuyuan19)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Course instructors and teaching assistants
- PCAWG Consortium for providing the data
- Team members for collaboration and insights

---

**Note**: This is a course project for educational purposes. The data and methods are based on published research.

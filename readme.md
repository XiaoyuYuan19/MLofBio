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

### Performance Visualization

![Model Comparison](results/figures/model_comparison.png)
*Figure 1: Comparison of model accuracies across different datasets (from Summary of Methods slide)*

![Confusion Matrix - DNN](results/figures/confusion_matrix_dnn.png)
*Figure 2: Confusion matrix for Deep Neural Network on combined WGS PCAWG dataset (22 cancer types, 79.46% accuracy)*

![Feature Importance](results/figures/feature_importance_gb.png)
*Figure 3: Feature importance analysis showing C>T mutation channel as most predictive*

## 📊 Dataset

### Data Source
Data from [Alexandrov et al. (2020) Nature](https://www.nature.com/articles/s41586-020-1943-3) - "The repertoire of mutational signatures in human cancer"

**Note**: The original datasets are not included in this repository due to size constraints. The data structure is described below for reproduction purposes.

### Data Structure

```
project_data/
├── catalogs/
│   ├── WGS/
│   │   ├── WGS_PCAWG.96.csv          # 96 channels × 2780 samples
│   │   └── WGS_Other.96.csv
│   └── WES/
│       ├── WES_TCGA.96.csv           # 96 channels × N samples
│       └── WES_Other.96.csv
└── activities/
    ├── WGS/
    │   ├── WGS_PCAWG.activities.csv  # 65 SBS signatures × 2780 samples
    │   └── WGS_Other.activities.csv
    └── WES/
        ├── WES_TCGA.activities.csv   # 65 SBS signatures × N samples
        └── WES_Other.activities.csv
```

#### Catalog File Format (e.g., WGS_PCAWG.96.csv)
```
Mutation type | Trinucleotide | Cancer-Type1::Sample1 | Cancer-Type1::Sample2 | ...
C>A          | ACA           | 245                   | 187                   | ...
C>A          | ACC           | 312                   | 298                   | ...
...
```
- Rows: 96 mutation channels (6 base mutations × 16 trinucleotide contexts)
- Columns: Sample IDs in format "CancerType::SampleID"
- Values: Mutation counts

#### Activity File Format (e.g., WGS_PCAWG.activities.csv)
```
Cancer Types | Sample Names | Accuracy | SBS1  | SBS2  | ... | SBS94
Biliary-AdenoCA | SP117655  | 0.95     | 2456  | 123   | ... | 0
...
```
- Rows: Individual samples
- Columns: Cancer type, sample name, reconstruction accuracy, 65 SBS signature activities

### Data Availability

The original PCAWG data can be accessed through:
1. **ICGC Data Portal**: [https://dcc.icgc.org/pcawg](https://dcc.icgc.org/pcawg)
2. **Synapse**: Requires registration and data access approval
3. **Supplementary materials** from [Alexandrov et al. 2020](https://www.nature.com/articles/s41586-020-1943-3)

For TCGA data:
- **GDC Data Portal**: [https://portal.gdc.cancer.gov/](https://portal.gdc.cancer.gov/)

### Cancer Types Coverage

- **WGS PCAWG**: 37 cancer types (2,780 samples)
- **WES TCGA**: Multiple cancer types with varying sample sizes
- After filtering (≥5 samples): 22 major cancer classes

## 🏗️ Project Structure

```
CancerTypeNet/
├── README.md
├── requirements.txt
├── LICENSE
├── notebooks/
│   ├── 01_CancerTypeNet_Main.ipynb          # Main analysis pipeline
│   └── 02_Data_Exploration.ipynb            # EDA and visualization
├── src/
│   ├── data_preprocessing.py                # Data loading and preprocessing
│   ├── models.py                            # Model architectures
│   └── evaluation.py                        # Evaluation metrics and visualization
├── project_data/                            # Data directory (not included in repo)
│   ├── catalogs/
│   └── activities/
└── results/
    ├── figures/                             # Generated figures
    │   ├── model_comparison.png
    │   ├── confusion_matrix_*.png
    │   └── feature_importance_*.png
    └── models/                              # Saved model checkpoints
        ├── dnn_best.pth
        ├── cnn_best.pth
        └── model_metrics.json
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

# Create necessary directories
mkdir -p project_data/{catalogs/{WGS,WES},activities/{WGS,WES}}
mkdir -p results/{figures,models}
```

### Data Preparation

1. Download the PCAWG/TCGA datasets following the instructions in the **Data Availability** section
2. Place the CSV files in the corresponding directories under `project_data/`
3. Verify data structure matches the format described above

### Quick Start

```python
# Load and preprocess data
from src.data_preprocessing import load_pcawg_data

catalogs, activities, cancer_types = load_pcawg_data()

# Train a model
from src.models import train_dnn

model, accuracy = train_dnn(
    X=catalogs, 
    y=cancer_types,
    epochs=2000,
    batch_size=32,
    learning_rate=0.001
)

print(f"Test Accuracy: {accuracy:.2%}")
```

### Running the Full Pipeline

```bash
# Run the main analysis notebook
jupyter notebook notebooks/01_CancerTypeNet_Main.ipynb
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

## 👥 Team Members

This project was completed as a collaborative team effort with equal contributions from all members:

- **Team Member 1**
- **Xiaoyu Yuan** - [xiaoyuyuan19@gmail.com](mailto:xiaoyuyuan19@gmail.com)
- **Team Member 3**
- **Team Member 4**

*All team members contributed equally to data analysis, model implementation, and manuscript preparation.*

## 🎓 Course Information

**Course**: Machine Learning in Molecular Biosciences (LSI31003)  
**Institution**: University of Helsinki  
**Date**: March 2024  
**Type**: Group Course Project

## 📝 Citation

If you use this work, please cite:

```bibtex
@misc{cancertypenet2024,
  title={CancerTypeNet: Tumor Type Classification from Mutational Signatures},
  author={Team Members and Yuan, Xiaoyu},
  year={2024},
  institution={University of Helsinki},
  note={Course project for Machine Learning in Molecular Biosciences}
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
- [GDC Data Portal](https://portal.gdc.cancer.gov/)

## 📧 Contact

**Xiaoyu Yuan**  
Email: xiaoyuyuan19@gmail.com  
Portfolio: [xiaoyuyuan19.github.io/portfolio](https://xiaoyuyuan19.github.io/portfolio/)  
LinkedIn: [linkedin.com/in/xiaoyuyuan19](http://www.linkedin.com/in/xiaoyuyuan19)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Course instructors and teaching assistants at University of Helsinki
- PCAWG Consortium for providing the mutational signature data
- All team members for their equal contributions and collaboration

---

**Note**: This is a course project for educational purposes. The data and methods are based on published research.

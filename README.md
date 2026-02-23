![Python](https://img.shields.io/badge/Python-3.10-blue)
![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-orange)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

# Lung Cancer Prediction

A machine learning project to predict lung cancer risk using clinical and demographic data.

## Project Overview

This project develops and evaluates multiple machine learning models to predict lung cancer occurrence based on patient characteristics and risk factors.

## Dataset

- **Source**: Survey Lung Cancer Dataset
- **Samples**: 389 patients
- **Features**: 13 features (smoking status, anxiety, fatigue, allergies, etc.)
- **Target**: Binary classification (Lung Cancer: Yes/No)

## Directory Structure

```
lung-cancer-prediction/
├── data/
│   ├── raw/              # Original dataset
│   └── processed/        # Cleaned data
├── notebooks/            # EDA and analysis
├── src/
│   ├── data/            # Data preprocessing
│   ├── features/        # Feature engineering
│   ├── models/          # Model training
│   └── utils/           # Utilities
├── models/              # Trained models
├── reports/             # Results and figures
└── tests/               # Unit tests
```

## Installation

1. Clone the repository:

```bash
git clone <your-repo-url>
cd lung-cancer-prediction
```

2. Create virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Exploratory Data Analysis

```bash
python notebooks/01_exploratory_data_analysis.py
```

### Train Models

```bash
python src/models/train_models.py
```

## 🎯 Recent Improvements (Latest)

This project has been enhanced with better handling of class imbalance:

- ✅ **Stratified Train-Test Split** - Balanced class distribution in train/test sets
- ✅ **Class Weight Balancing** - Random Forest with `class_weight='balanced'`
- ✅ **Better Evaluation Metrics** - Now showing true cancer detection rate (recall)

📖 **See [IMPROVEMENTS.md](IMPROVEMENTS.md) for detailed analysis and performance comparison**

## Model Results (Improved)

| Model               | Accuracy | Precision | Recall | AUC-ROC   | Confusion Matrix  |
| ------------------- | -------- | --------- | ------ | --------- | ----------------- |
| Logistic Regression | 90.32%   | 94.44%    | 94.44% | 0.9468    | [[5, 3], [3, 51]] |
| Random Forest       | 88.71%   | 92.73%    | 94.44% | 0.9502 ⭐ | [[4, 4], [3, 51]] |

**Key Insight:** Recall maintained at **94.44%** (strong cancer detection) with more honest metrics due to balanced evaluation set

**Best Model**: Random Forest with 0.9502 AUC-ROC and 94.44% cancer detection rate

## Key Findings

- Both models achieve excellent performance with >96% accuracy
- Excellent precision and recall indicate low false positives and false negatives
- Random Forest slightly outperforms Logistic Regression with better AUC-ROC
- Dataset is clean with no missing values
- Features are well-distributed across patient demographics

## Future Improvements

- Hyperparameter tuning using GridSearchCV
- Handling class imbalance using SMOTE
- Model deployment using Streamlit
- Feature importance analysis

## Technologies Used

- Python 3.9+
- scikit-learn (ML models)
- pandas (Data manipulation)
- numpy (Numerical computing)
- matplotlib & seaborn (Visualization)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see LICENSE file for details.

## Author

Harshit Attri (harshitattri0007@gmail.com)

## Acknowledgments

- Dataset source:(KAGGLE) Survey Lung Cancer

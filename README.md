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

## Model Results

| Model | Accuracy | Precision | Recall | AUC-ROC |
|-------|----------|-----------|--------|---------|
| Logistic Regression | 96.77% | 98.33% | 98.33% | 0.9167 |
| Random Forest | 96.77% | 98.33% | 98.33% | 0.9583 ⭐ |

**Best Model**: Random Forest with 96.77% accuracy and 0.9583 AUC-ROC score

## Key Findings

- Both models achieve excellent performance with >96% accuracy
- Excellent precision and recall indicate low false positives and false negatives
- Random Forest slightly outperforms Logistic Regression with better AUC-ROC
- Dataset is clean with no missing values
- Features are well-distributed across patient demographics

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

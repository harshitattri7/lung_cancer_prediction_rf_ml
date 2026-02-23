# Model Deployment Guide

## 📦 Deployed Models

This repository includes pre-trained, production-ready models for lung cancer prediction.

### Model Files

- **`models/model.pkl`** - Random Forest classifier (Best performing model)
- **`models/scaler.pkl`** - StandardScaler for feature normalization

---

## 🚀 Quick Start - Using the Deployed Model

### Step 1: Load the Model

```python
import pickle
import pandas as pd
import numpy as np

# Load the pre-trained model and scaler
with open('models/model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
```

### Step 2: Prepare Your Data

```python
# Load your data (must have the same 13 features)
# Features: GENDER, AGE, SMOKING, YELLOW_FINGERS, ANXIETY,
#           PEER_PRESSURE, CHRONIC_DISEASE, FATIGUE, ALLERGY, WHEEZING,
#           ALCOHOL_CONSUMING, COUGHING, SHORTNESS_OF_BREATH

data = pd.read_csv('your_data.csv')

# Encode categorical variables (same as training)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

for col in data.columns:
    if data[col].dtype == 'object':
        data[col] = le.fit_transform(data[col])
```

### Step 3: Make Predictions

```python
# Scale the features
X_scaled = scaler.transform(data)

# Make predictions
predictions = model.predict(X_scaled)
probabilities = model.predict_proba(X_scaled)

# Print results
for i, pred in enumerate(predictions):
    lung_cancer = "YES" if pred == 1 else "NO"
    confidence = probabilities[i][pred] * 100
    print(f"Patient {i+1}: Lung Cancer Risk = {lung_cancer} (Confidence: {confidence:.2f}%)")
```

---

## 📊 Model Performance

**Random Forest Classifier**

- **Accuracy**: 88.71%
- **Precision**: 92.73% (Few false positives)
- **Recall**: 94.44% (Detects 94% of cancer cases)
- **AUC-ROC**: 0.9502 (Excellent discrimination)

### Confusion Matrix

```
               Predicted
               No Cancer  Cancer
Actual No        4         4
       Cancer    3        51
```

---

## ⚙️ Model Details

### Algorithm

- **Type**: Random Forest Classifier
- **Number of Trees**: 100
- **Random State**: 42 (for reproducibility)
- **Class Weight**: Balanced (handles imbalanced dataset)

### Features (13 Total)

All features are normalized to [0, 1] range using StandardScaler

### Training Data

- **Dataset**: Survey Lung Cancer Dataset
- **Samples**: 389 patients
- **Class Distribution**: 270 positive cases, 39 negative cases
- **Train/Test Split**: 80/20 with stratification

---

## 🔧 Advanced Usage

### Get Prediction Probabilities

```python
# Get probability for each class
predictions_proba = model.predict_proba(X_scaled)

# predictions_proba[:, 0] = Probability of No Cancer
# predictions_proba[:, 1] = Probability of Cancer

for i, probs in enumerate(predictions_proba):
    print(f"Patient {i+1}:")
    print(f"  Probability of No Cancer: {probs[0]:.4f}")
    print(f"  Probability of Cancer: {probs[1]:.4f}")
```

### Feature Importance

```python
# Get feature importance scores
feature_importance = model.feature_importances_

# Assuming you have feature names
feature_names = ['GENDER', 'AGE', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY',
                 'PEER_PRESSURE', 'CHRONIC_DISEASE', 'FATIGUE', 'ALLERGY',
                 'WHEEZING', 'ALCOHOL_CONSUMING', 'COUGHING', 'SHORTNESS_OF_BREATH']

# Display importance
for name, importance in zip(feature_names, feature_importance):
    print(f"{name}: {importance:.4f}")
```

---

## ⚠️ Important Notes

1. **Input Features Required**: Model expects exactly 13 features in specific order
2. **Data Preprocessing**:
   - Categorical variables must be encoded to numeric
   - All features must be scaled using the provided scaler
3. **Class Imbalance**: Model is trained with balanced class weights to handle imbalanced data
4. **Prediction Format**:
   - 0 = No Cancer
   - 1 = Cancer Detected

---

## 🔄 Retraining the Model

To retrain the model with new data:

```bash
# Update your data in data/raw/survey lung cancer.csv

# Run the training script
python3 src/models/train_models.py

# This will:
# 1. Load and preprocess data
# 2. Train both Logistic Regression and Random Forest
# 3. Evaluate model performance
# 4. Save the best model to models/model.pkl
# 5. Save the scaler to models/scaler.pkl
```

---

## 📁 Files in This Directory

```
models/
├── model.pkl          # Trained Random Forest model
├── scaler.pkl         # Feature StandardScaler
└── .gitkeep          # Placeholder for git
```

---

## 🐛 Troubleshooting

### Error: "No module named 'sklearn'"

```bash
pip install scikit-learn
```

### Error: "Feature mismatch"

- Ensure your data has exactly 13 features
- Ensure features are in the correct order
- Ensure all features are numeric (encoded if necessary)

### Error: "Shape mismatch during scaling"

- Check that all 13 features are present
- Check that no features are missing or extra

---

## 📝 Citation & References

**Dataset**: Survey Lung Cancer Dataset
**Libraries**: scikit-learn, pandas, numpy

---

## 📧 Support

For questions or issues with the deployed model, please refer to the main [README.md](README.md) or create an issue on GitHub.

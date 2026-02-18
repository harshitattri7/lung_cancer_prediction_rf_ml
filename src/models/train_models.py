"""Model training script"""
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

class ModelTrainer:
    def __init__(self):
        self.models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
        }
        self.results = {}
    
    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        """Train all models and evaluate"""
        print("\n=== Training Models ===\n")
        
        for name, model in self.models.items():
            print(f"Training {name}...")
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_pred_proba)
            
            self.results[name] = {
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'AUC-ROC': auc
            }
            
            print(f"✓ {name} - Accuracy: {accuracy:.4f}, AUC: {auc:.4f}\n")
    
    def display_results(self):
        """Display results in a table"""
        results_df = pd.DataFrame(self.results).T
        print("\n=== Model Results ===")
        print(results_df)
        return results_df

if __name__ == "__main__":
    # Load data
    df = pd.read_csv('data/raw/survey lung cancer.csv')
    
    # Encode categorical variables
    le = LabelEncoder()
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = le.fit_transform(df[col])
    
    X = df.drop('LUNG_CANCER', axis=1)
    y = df['LUNG_CANCER']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train models
    trainer = ModelTrainer()
    trainer.train_and_evaluate(X_train_scaled, X_test_scaled, y_train, y_test)
    trainer.display_results()
    
    print("\n✓ Model training complete!")

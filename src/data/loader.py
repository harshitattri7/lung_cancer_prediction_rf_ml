import pandas as pd
from pathlib import Path

class DataLoader:
    def __init__(self, data_path):
        self.data_path = Path(data_path)
    
    def load_csv(self, filename):
        """Load CSV file"""
        file_path = self.data_path / filename
        return pd.read_csv(file_path)
    
    def load_data(self, filename):
        """Load various file formats"""
        file_path = self.data_path / filename
        
        if filename.endswith('.csv'):
            return pd.read_csv(file_path)
        elif filename.endswith(('.xlsx', '.xls')):
            return pd.read_excel(file_path)
        elif filename.endswith('.parquet'):
            return pd.read_parquet(file_path)

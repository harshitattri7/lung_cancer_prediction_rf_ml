"""Data processing script - transforms raw data to processed data"""
import sys
sys.path.insert(0, '..')

from src.data.loader import DataLoader
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

def main():
    logger.info("Starting data processing...")
    # Load raw data
    loader = DataLoader('data/raw')
    # Add your preprocessing logic here
    logger.info("Data processing completed!")

if __name__ == "__main__":
    main()

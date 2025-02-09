import numpy as np
import pandas as pd
from PIL import Image
import os

class DataProcessor:
    def __init__(self, metadata_path):
        """
        Initialize data processor
        
        Args:
            metadata_path: Path to metadata CSV file
        """
        self.metadata_path = metadata_path
        
    def load_metadata(self):
        """Load metadata from CSV file"""
        print(f"Loading metadata from {self.metadata_path}")
        metadata = pd.read_csv(self.metadata_path)
        headers = metadata.columns.tolist()
        
        print(f"Loaded metadata shape: {metadata.shape}")
        print(f"Headers: {headers}")
        
        return metadata, headers
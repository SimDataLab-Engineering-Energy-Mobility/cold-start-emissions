import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

import os
import sys
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException


@dataclass
class DataPaths:
    """stores paths to data files"""
    X_train: str = os.path.join("artifacts", "X_train.npy")
    X_test: str = os.path.join("artifacts", "X_test.npy")
    y_train: str = os.path.join("artifacts", "y_train.npy")
    y_test: str = os.path.join("artifacts", "y_test.npy")

class DataPreprocessor:
    """Handles data loading, data normalization, and sequence processing"""
    def __init__(self, paths: DataPaths=DataPaths()):
        self.paths = paths
        self.X_scaler = StandardScaler()
        self.y_scaler = StandardScaler()
        
    def load_raw_data(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load raw data from files"""
        try:
            logging.info("Data loaded initiated")
            # Load data from .npy files
            X_train = np.load(self.paths.X_train)
            X_test = np.load(self.paths.X_test)
            y_train = np.load(self.paths.y_train)
            y_test = np.load(self.paths.y_test)
            logging.info("Data loaded successfully")
            return  (X_train, X_test, y_train, y_test)
        except Exception as e:
            raise CustomException(e, sys) 
    
    def normalize_data(self, X_train: np.ndarray, X_test: np.ndarray, 
                       y_train: np.ndarray, y_test: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Normalize the data using StandardScaler"""
        try:
            logging.info("Data normalization initiated")
            # Normalize X data
            X_train_scaled = self.x_scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1]))
            X_test_scaled = self.x_scaler.transform(X_test.reshape(-1, X_test.shape[-1]))
        
            # Normalize y data
            y_train_scaled = self.y_scaler.fit_transform(y_train.reshape(-1, y_train.shape[-1]))
            y_test_scaled = self.y_scaler.transform(y_test.reshape(-1, y_test.shape[-1]))
            
            # Reshape back to original shape
            X_train_scaled = X_train_scaled.reshape(X_train.shape)
            X_test_scaled = X_test_scaled.reshape(X_test.shape)
            y_train_scaled = y_train_scaled.reshape(y_train.shape)
            y_test_scaled = y_test_scaled.reshape(y_test.shape)
            logging.info("Data normalization completed")
            
            return (X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled)
        except Exception as e:
            raise CustomException(e, sys)
        
    @staticmethod
    def create_sequences(input_array: np.ndarray, chunk_start: int = 0, window_size: int = 80, step: int = 5) -> np.ndarray:
        """Convert time series data to sequences for seq2seq model"""
        logging.info("Creating sequences initiated")
        sequences = []
        for sample in input_array:
            sample_sequences = [
                sample[j:j+window_size]
                for j in range(chunk_start, len(sample) - window_size, step)
            ]
            sequences.append(sample_sequences)
        logging.info("Sequences created successfully")
        return np.array([np.array(seq) for seq in sequences])
    
    def data_processing(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load and process data"""
        try:
            logging.info("Data processing initiated")
            # Load raw data
            X_train, X_test, y_train, y_test = self.load_raw_data()
            
            # Normalize data
            X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = self.normalize_data(X_train, X_test, y_train, y_test)
            
            # Create sequences
            X_train_seq = self.create_sequences(X_train_scaled)
            X_test_seq = self.create_sequences(X_test_scaled)
            y_train_seq = self.create_sequences(y_train_scaled)
            y_test_seq = self.create_sequences(y_test_scaled)
            logging.info("Data processing completed")
            # Save processed data
            return (X_train_seq, X_test_seq, y_train_seq, y_test_seq)
        except Exception as e:
            raise CustomException(e, sys)
        

# usage
if __name__ == "__main__":
    # initialize the data preprocessor
    data_preprocessor = DataPreprocessor()
    # load the raw data
    X_train, X_test, y_train, y_test = data_preprocessor.load_raw_data()
        

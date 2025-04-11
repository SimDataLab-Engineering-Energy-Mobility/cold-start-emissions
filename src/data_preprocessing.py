# Copyright (C) 2025 Karlsruhe Institute of Technology (KIT)

# Scientific Computing Center (SCC), Department of Scientific Computing and Mathematics

# Authors: Manoj Mangipudi, Jordan A. Denev

# Licensed under the GNU General Public License v3.0

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from typing import Tuple
import os
import sys
from dataclasses import dataclass
from logger import logging
from exception import CustomException


@dataclass
class DataPaths:
    """stores paths to data files"""
    X_train: str = os.path.join("sample_training_data", "X_train.npy")
    X_test: str = os.path.join("sample_training_data", "X_test.npy")
    y_train: str = os.path.join("sample_training_data", "y_train.npy")
    y_test: str = os.path.join("sample_training_data", "y_test.npy")

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
            X_train_scaled = self.X_scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1]))
            X_test_scaled = self.X_scaler.transform(X_test.reshape(-1, X_test.shape[-1]))
        
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
            X_train_seq = self.create_sequences(X_train_scaled, chunk_start =0, window_size = 80)
            X_test_seq = self.create_sequences(X_test_scaled, chunk_start =0, window_size = 80)
            y_train_seq = self.create_sequences(y_train_scaled, chunk_start =78, window_size = 5)
            y_test_seq = self.create_sequences(y_test_scaled, chunk_start =78, window_size = 5)
            logging.info("X_train_seq shape: %s", X_train_seq.shape)
            logging.info("X_test_seq shape: %s", X_test_seq.shape)
            logging.info("y_train_seq shape: %s", y_train_seq.shape)
            logging.info("y_test_seq shape: %s", y_test_seq.shape)
            logging.info("Data processing completed")
            # Save processed data
            return (X_train_seq, X_test_seq, y_train_seq, y_test_seq)
        except Exception as e:
            raise CustomException(e, sys)
    

class PytorchSequeceDataset(Dataset):
    """PyTorch Dataset for sequence data"""
    def __init__(self, input_data: np.ndarray, target_data: np.ndarray):
        # reshape to (total_sequences, sequence_length, features)
        
        self.input_data = torch.tensor(
            input_data.reshape(-1, *input_data.shape[-2:]),
                               dtype=torch.float32)
        self.target_data = torch.tensor(
            target_data.reshape(-1, *target_data.shape[-2:]),
            dtype=torch.float32)     
    
    def __len__(self) -> int:
        return len(self.input_data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.input_data[idx], self.target_data[idx] 

class DataLoaderFact:
    """Creates data loaders for training and evaluation"""
    
    @staticmethod
    def create_loaders(
        x_train: np.ndarray, x_test: np.ndarray,
        y_train: np.ndarray, y_test: np.ndarray,
        batch_size: int = 256) -> Tuple[DataLoader, DataLoader]:
        
        """Create train and test data loaders"""
        logging.info("Creating PyTorch dataset initiated")
        logging.info("Creating data loaders initiated")
        # Create datasets
        train_dataset = PytorchSequeceDataset(x_train, y_train)
        test_dataset = PytorchSequeceDataset(x_test, y_test)
        
        return (
            DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
            DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
        )


# usage

if __name__ == "__main__":
    # initialize the data preprocessor
    data_preprocessor = DataPreprocessor()
    # load the raw data
    X_train, X_test, y_train, y_test = data_preprocessor.data_processing()  
    
    # Create data loaders
    train_loader, test_loader = DataLoaderFact.create_loaders(X_train, X_test, y_train, y_test)
    #print(f"Total training batches: {len(train_loader)}")
    #print(f"Total test batches: {len(test_loader)}")
    # Should have multiple batches unless batch_size >= dataset size
    print(f"Dataset size: {len(train_loader.dataset)}")
    print(f"Batch size: {train_loader.batch_size}")
    print(f"Calculated batches: {len(train_loader.dataset)/train_loader.batch_size:.1f}")
    
    # Print shapes for verification
    #sample_input, sample_target = next(iter(train_loader))
    #print(f"Train loader - Input shape: {sample_input.shape}, Target shape: {sample_target.shape}")
    
    #test_input, test_target = next(iter(test_loader))
    #print(f"Test loader - Input shape: {test_input.shape}, Target shape: {test_target.shape}")     
 

"""
    Complete training pipeline with visualization and model saving
    
    Args:
        train_loader: Training data loader
        test_loader: Validation data loader
        model_params: Dictionary of model hyperparameters
        save_dir: Directory to save trained models
        
    Returns:
        tuple: (trained_model, train_loss_history, val_loss_history)
"""


"""def train_model():
    # Initialize components
    preprocessor = DataPreprocessor()
    
    # Prepare data
    x_train, x_test, y_train, y_test = preprocessor.data_processing()
    
    # Create data loaders
    train_loader, test_loader = DataLoaderFact.create_loaders(x_train, x_test, y_train, y_test)
    
    # Inspect batches before training
    inspect_batches(train_loader, num_batches=2)
    
    # initialize the model
"""  
    

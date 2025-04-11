# Copyright (C) 2025 Karlsruhe Institute of Technology (KIT)

# Scientific Computing Center (SCC), Department of Scientific Computing and Mathematics

# Authors: Manoj Mangipudi, Jordan A. Denev

# Licensed under the GNU General Public License v3.0

import os
import time
import torch
import glob
import torch.nn as nn
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from datetime import datetime

from architecture import Encoder, Decoder, Seq2Seq
from data_preprocessing import DataPreprocessor
from data_preprocessing import DataLoaderFact
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error

from logger import logging
from exception import CustomException
from typing import Tuple, List  
from utils import round_numbers, range_with_floats, get_formatter


def inspect_batches(train_loader: DataLoader, num_batches: int = 2):
    """Prinshape and sample data from the first few batches"""
    print("\n" + "="*50)
    print(f"Inspecting {num_batches} batches from train_loader")
    print(f"Total batches: {len(train_loader)}")
    print(f"Batch size: {train_loader.batch_size}")
    print("="*50 + "\n")

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        if batch_idx >= num_batches:
            break

        print(f"Batch {batch_idx + 1}:")
        print(f"  Input shape: {inputs.shape} (batch_size, seq_len, features)")
        print(f"  Target shape: {targets.shape}")
        print(f"  Input dtype: {inputs.dtype}, Target dtype: {targets.dtype}")
        print(f"  Input range: [{inputs.min():.4f}, {inputs.max():.4f}]")
        print(f"  Target range: [{targets.min():.4f}, {targets.max():.4f}]")
        
        # Print first sequence's first and last timestep
        print("\n  Sample Sequence (First in batch):")
        print("  Input[0, 0]:", inputs[0, 0].numpy().round(4))  # First timestep
        print("  Input[0, -1]:", inputs[0, -1].numpy().round(4)) # Last timestep
        print("  Target[0, 0]:", targets[0, 0].numpy().round(4))
        print("-"*40 + "\n")
        

def train_model(
    train_loader: DataLoader,
    test_loader: DataLoader,
    model_params: dict = None,
    save_dir: str = 'saved_models', 
    fig_dir: str = 'results'
) -> Tuple[nn.Module, List[float], List[float]]:
    """
    Complete training pipeline with automatic figure saving
    
    Args:
        train_loader: Training data loader
        test_loader: Validation data loader
        model_params: Dictionary of model hyperparameters
        save_dir: Directory to save trained models
        fig_dir: Directory to save training figures
        
    Returns:
        tuple: (trained_model, train_loss_history, val_loss_history)
    """
    # Default hyperparameters
    default_params = {
        'input_dim': 7,
        'output_dim': 3,
        'hidden_dim': 128,
        'num_layers': 2,
        'dropout': 0.2,
        'learning_rate': 0.0001,
        'num_epochs': 300,
        'output_length': 5
    }
    
    params = {**default_params, **(model_params or {})}
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize model
    logging.info("Initializing model...")
    encoder = Encoder(
        params['input_dim'],
        params['hidden_dim'],
        params['num_layers'],
        params['dropout']
    ).to(device)
    
    decoder = Decoder(
        params['output_dim'],
        params['hidden_dim'],
        params['num_layers'],
        params['dropout']
    ).to(device)
    
    model = Seq2Seq(
        encoder,
        decoder,
        device,
        params['output_length']
    ).to(device)
    logging.info("Model initialized.")
    logging.info("initialized model with parameters:")
    # Loss and optimizer
    criterion = nn.HuberLoss(delta=0.25)
    optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
    
    # Training tracking
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    # Create figure directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fig_path = os.path.join(fig_dir, timestamp)
    os.makedirs(fig_path, exist_ok=True)
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    logging.info(f"Model checkpoints will be saved to: {save_dir}")
    logging.info(f"Training figures will be saved to: {fig_path}")
    
    # Training loop
    start_time = time.time()
    logging.info("Starting training...")
    for epoch in range(params['num_epochs']):
        # Training phase
        model.train()
        epoch_train_loss = 0
        for src, trg in train_loader:
            src, trg = src.to(device), trg.to(device)
            
            optimizer.zero_grad()
            output = model(src)
            loss = criterion(output, trg)
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item()
        
        # Validation phase
        model.eval()
        epoch_val_loss = 0
        all_preds, all_actuals = [], []
        with torch.no_grad():
            for src, trg in test_loader:
                src, trg = src.to(device), trg.to(device)
                output = model(src)
                loss = criterion(output, trg)
                epoch_val_loss += loss.item()
                
                # Store for visualization
                if epoch == 0 or (epoch + 1) % 50 == 0:
                    all_preds.append(output.cpu())
                    all_actuals.append(trg.cpu())
                    
        # Calculate epoch metrics
        epoch_train_loss /= len(train_loader)
        epoch_val_loss /= len(test_loader)
        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        print(f"Epoch [{epoch+1}/{params['num_epochs']}], Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}")
        logging.info(f"Epoch [{epoch+1}/{params['num_epochs']}], Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}")
        # Save model
        save_path = os.path.join(save_dir, f'epoch_{epoch+1}.pth')
        torch.save(model.state_dict(), save_path)
    
    # After training, find the epoch with the minimum validation loss
    best_epoch = val_losses.index(min(val_losses)) + 1  # +1 because we start from epoch 1
    print(f'Best model at epoch {best_epoch} with training loss {min(val_losses):.4f}')
    logging.info(f'Best model at epoch {best_epoch} with training loss {min(val_losses):.4f}')
    
    # Save the best model
    # Delete other model checkpoints
    for epoch in range(1, params['num_epochs'] + 1):
        if epoch != best_epoch:
            file_path = os.path.join(save_dir, f'epoch_{epoch}.pth')
            if os.path.exists(file_path):
                os.remove(file_path)

    print(f"Kept only model from epoch {best_epoch}. Others deleted.")    
    logging.info(f"Kept only model from epoch {best_epoch}. Others deleted.")        
    
    # Finalization
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time//60:.0f}m {training_time%60:.2f}s")
    print(f"Best validation loss: {best_val_loss:.4f}")
    
    # Plot loss curves
    logging.info("Plotting loss curves...")
    plot_loss_curves(
        train_losses, 
        val_losses,
        save_path=os.path.join(fig_path, 'loss_curves.png')
    )
    
    return model, train_losses, val_losses
    
def plot_loss_curves(train_losses: List[float], val_losses: List[float], save_path: str = None) -> None:
    """Plot training and validation loss curves"""
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curves')
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()
    

def visualize_predictions(
    preds: torch.Tensor,
    targets: torch.Tensor,
    epoch: int,
    n_samples: int = 3,
    save_path: str = None
) -> plt.Figure:
    """Visualize model predictions vs actual values"""
    plt.figure(figsize=(15, 5))
    for i in range(min(n_samples, preds.size(0))):
        plt.subplot(1, n_samples, i+1)
        plt.plot(targets[i, :, 0], 'b-', label='Actual')
        plt.plot(preds[i, :, 0], 'r--', label='Predicted')
        plt.title(f'Epoch {epoch} - Sample {i+1}')
        plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()
    
    return plt.gcf()

# ======================= Model Functions =======================

def load_model(model_path, input_dim, output_dim, hidden_dim, num_layers, dropout, output_length, device):
    """
    Load a pre-trained sequence-to-sequence model.
    """
    logging.info('load the pretrained model')
    encoder = Encoder(input_dim, hidden_dim, num_layers, dropout).to(device)
    decoder = Decoder(output_dim, hidden_dim, num_layers, dropout).to(device)
    model = Seq2Seq(encoder, decoder, device, output_length).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def evaluate_model(model, data_loader, device):
    """
    Run model inference on the provided data loader.
    """
    logging.info('Evaluating model start...')
    all_preds, all_actuals = [], []
    for src, trg in data_loader:
        src, trg = src.to(device), trg.to(device)
        with torch.no_grad():
            output = model(src)
        all_preds.append(output.cpu().numpy())
        all_actuals.append(trg.cpu().numpy())
    logging.info('Evaluating model end...')
    # Concatenate all predictions and actuals
    return np.concatenate(all_preds), np.concatenate(all_actuals)

# ======================= Data Processing Functions =======================
def reshape_data(data, target_shape):
    """Reshape data to the desired target shape."""
    return data.reshape(target_shape)

def inverse_transform(data, scaler):
    """Inverse transform data using the given scaler."""
    return scaler.inverse_transform(data)

# ======================= Evaluation Metrics =======================

def calculate_metrics(actuals, predictions, steps, interval=2420):
    """
    Compute R2, RMSE, and MAE metrics in intervals.
    """
    logging.info('Calculating metrics start...')
    metrics = {'R2': [], 'RMSE': [], 'MAE': []}
    for step in range(0, steps, interval):
        actual_slice = actuals[step:step + interval]
        pred_slice = predictions[step:step + interval]

        metrics['R2'].append([r2_score(actual_slice[:, i], pred_slice[:, i]) for i in range(actual_slice.shape[1])])
        metrics['RMSE'].append([root_mean_squared_error(actual_slice[:, i], pred_slice[:, i]) for i in range(actual_slice.shape[1])])
        metrics['MAE'].append([mean_absolute_error(actual_slice[:, i], pred_slice[:, i]) for i in range(actual_slice.shape[1])])

    # Round metrics
    return {key: [round_numbers(val) for val in values] for key, values in metrics.items()}

# ======================= Plotting Functions =======================

# Plotting function
def plot_subplot(ax, time, y_true, y_pred, title, ylabel, r2, rmse, formatter):
    """_summary_

    Args:
        ax (): _description_
        time (_type_): _description_
        y_true (_type_): _torcharray_
        y_pred (_type_): _torcharray_
        title (_type_): _str_
        ylabel (_type_): _str_
        r2 (_type_): _dictionary_
        rmse (_type_): _dictionary_
        formatter (_type_): _description_
    """
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    
    ax.plot(time, y_true, linewidth=1.8, color='b', label='Original')
    ax.plot(time, y_pred, linewidth=1.8, color='r', label='Predicted')
    
    ax.set_title(title, fontsize=22, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=20)
    ax.text(0.05, 0.95,
            f"$\\mathrm{{R2\\text{{-}}score}} = {r2:.2f}$\n$\\mathrm{{RMSE\\text{{-}}score}} = {rmse:.2f}$",
            transform=ax.transAxes, fontsize=17, verticalalignment='top', bbox=props)
    
    ax.legend(loc='upper right')
    ax.grid(linestyle='dotted')
    ax.spines['bottom'].set_linewidth(1.8)
    ax.spines['top'].set_linewidth(1.8)
    ax.spines['left'].set_linewidth(1.8)
    ax.spines['right'].set_linewidth(1.8)
    ax.yaxis.set_major_formatter(formatter)
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)
    ax.set_ylim(bottom=0)

def create_plots(data_indices, titles, output_filename, actuals_final, preds_final, metrics, time_seconds, formatter_y):
    """Create and save plots for the given data indices and titles."""
    logging.info("Creating plots...")
    plt.rcParams['font.size'] = 18
    plt.rcParams['axes.titlepad'] = 20
    fig, axs = plt.subplots(3, 2, figsize=(18, 14))

    plot_data = []

    for idx, temp_label in zip(data_indices, titles):
        # NOx
        plot_data.append((axs[0, 0 if idx % 2 == 1 else 1], actuals_final[idx][:, 0], preds_final[idx][:, 0],
                          f'RDE {temp_label}', 'NOx [ppm]', metrics['R2'][idx][0], metrics['RMSE'][idx][0]))

        # CO
        plot_data.append((axs[1, 0 if idx % 2 == 1 else 1], actuals_final[idx][:, 1], preds_final[idx][:, 1],
                          f'RDE {temp_label}', 'CO [ppm]', metrics['R2'][idx][1], metrics['RMSE'][idx][1]))

        # UHC
        plot_data.append((axs[2, 0 if idx % 2 == 1 else 1], actuals_final[idx][:, 2], preds_final[idx][:, 2],
                          f'RDE {temp_label}', 'UHC [ppm]', metrics['R2'][idx][2], metrics['RMSE'][idx][2]))

    # Plot each subplot
    for ax, y_true, y_pred, title, ylabel, r2, rmse in plot_data:
        plot_subplot(ax, time_seconds, y_true, y_pred, title, ylabel, r2, rmse, formatter_y)

    plt.tight_layout()
    plt.savefig(os.path.join('results/plots', output_filename), dpi=300)
    plt.close(fig)  # Close the figure after saving to free memory


# ======================= Main Execution =======================

if __name__ == "__main__":
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    # initialize the data preprocessor
    data_preprocessor = DataPreprocessor()
    
    # load the raw data
    X_train, X_test, y_train, y_test = data_preprocessor.data_processing()  
    
    # Create data loaders
    train_loader, test_loader = DataLoaderFact.create_loaders(X_train, X_test, y_train, y_test)
    # Inspect batches before training
    inspect_batches(train_loader, num_batches=2)
    # initialize the model
    model_params = {
        'input_dim': 7,
        'output_dim': 3,
        'hidden_dim': 128,
        'num_layers': 2,
        'dropout': 0.2,
        'learning_rate': 0.0001,
        'num_epochs': 50,
        'output_length': 5
    } 
    # Train the model   
    model, train_losses, val_losses = train_model(
        train_loader,
        test_loader,
        model_params=model_params
    ) 
    
    # Evaluation phase
    # Load model
    model_paths = glob.glob("saved_models/*.pth")
    if not model_paths:
        raise FileNotFoundError("No saved model found in 'saved_models/' directory.")
    model = load_model(model_paths[0], model_params['input_dim'],
                       model_params['output_dim'], 
                       model_params['hidden_dim'],
                       model_params['num_layers'],
                       model_params['dropout'],
                       model_params['output_length'], device)
    # Evaluate model
    predictions, actuals = evaluate_model(model, test_loader, device)

    # Data Transformation
    sc = data_preprocessor.y_scaler
    actuals = inverse_transform(reshape_data(actuals, (-1, model_params['output_dim'])), sc)
    predictions = inverse_transform(reshape_data(predictions, (-1, model_params['output_dim'])), sc)
    print(f'shpae of actuals: {actuals.shape}')
    print(f'shpae of predictions: {predictions.shape}')
    
    # Reshape for metric calculation
    reshaped_for_metrics = (4, 2420, 3)
    actuals_final = reshape_data(actuals, reshaped_for_metrics)
    preds_final = reshape_data(predictions, reshaped_for_metrics)
    print(f"Shapes: actuals {actuals_final.shape}, predictions {preds_final.shape}")
    
    # Metrics calculation
    total_steps = actuals.shape[0]
    metrics = calculate_metrics(actuals, predictions, total_steps)

    # Print metrics
    gases = ['NO', 'CO', 'UHC']
    for idx, gas in enumerate(gases):
        print(f"\nMetrics for {gas}:")
        print(f"R2: { [val[idx] for val in metrics['R2']] }")
        print(f"RMSE: { [val[idx] for val in metrics['RMSE']] }")
        print(f"MAE: { [val[idx] for val in metrics['MAE']] }")
        logging.info(f"\nMetrics for {gas}:")
        logging.info(f"R2: { [val[idx] for val in metrics['R2']] }")
        logging.info(f"RMSE: { [val[idx] for val in metrics['RMSE']] }")
        logging.info(f"MAE: { [val[idx] for val in metrics['MAE']] }")

    # Plot results
    time_series = range_with_floats(0, 800, 0.2)[:2420]
    # Formatters
    formatter_y = get_formatter((-4, 4))
    logging.info("Creating plots... stored in results folder")
    create_plots(
        data_indices=[1, 0],
        titles=['-15$^\\circ$C', '-7$^\\circ$C'],
        output_filename='plots1.png',
        actuals_final=actuals_final,
        preds_final=preds_final,
        metrics=metrics,
        time_seconds=time_series,
        formatter_y=formatter_y
    )

    create_plots(
        data_indices=[2, 3],
        titles=['0$^\\circ$C', '23$^\\circ$C'],
        output_filename='plots2.png',
        actuals_final=actuals_final,
        preds_final=preds_final,
        metrics=metrics,
        time_seconds=time_series,
        formatter_y=formatter_y)

     
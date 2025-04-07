import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from datetime import datetime

from architecture import Encoder, Decoder, Seq2Seq
from data_preprocessing import DataPreprocessor
from data_preprocessing import DataLoaderFact
from torch.utils.data import Dataset, DataLoader

from logger import logging
from exception import CustomException
from typing import Tuple, List  


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
    
    # Loss and optimizer
    criterion = nn.HuberLoss()
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
    
    # Training loop
    start_time = time.time()
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
        
        # Save model checkpoints
        if (epoch + 1) % 10 == 0:
            save_path = os.path.join(save_dir, f'epoch_{epoch+1}.pth')
            torch.save(model.state_dict(), save_path)
            
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
            
        # Print progress
        print(f"Epoch [{epoch+1}/{params['num_epochs']}] "
              f"Train Loss: {epoch_train_loss:.4f}, "
              f"Val Loss: {epoch_val_loss:.4f}")
        
        # Visualize sample predictions periodically
        if epoch == 0 or (epoch + 1) % 50 == 0:
            visualize_predictions(
                torch.cat(all_preds),
                torch.cat(all_actuals),
                epoch=epoch+1,
                save_path=os.path.join(fig_path, f'predictions_epoch_{epoch+1}.png')
            )
            
    # Finalization
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time//60:.0f}m {training_time%60:.2f}s")
    print(f"Best validation loss: {best_val_loss:.4f}")
    
    # Plot loss curves
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
    
    

if __name__ == "__main__":
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
        'num_epochs': 300,
        'output_length': 5
    } 
    # Train the model   
    model, train_losses, val_losses = train_model(
        train_loader,
        test_loader,
        model_params=model_params
    ) 
    """# Save the model
    torch.save(model.state_dict(), 'saved_models/final_model.pth')  
    print("Model saved to 'saved_models/final_model.pth'")
    # Plot loss curves      
    plot_loss_curves(train_losses, val_losses)
    # Visualize predictions
    visualize_predictions(
        torch.cat(all_preds),
        torch.cat(all_actuals),
        epoch=model_params['num_epochs']
    )
    """
     
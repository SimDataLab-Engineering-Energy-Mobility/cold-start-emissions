import glob
import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error

from data_preprocessing import DataPreprocessor
from architecture import Seq2Seq, Encoder, Decoder
from data_preprocessing import DataLoaderFact
from logger import logging
from exception import CustomException

def round_numbers(lst, num_decimal_places=0):
    """Round numbers in a list to a specified number of decimal places."""
    return [round(num, num_decimal_places) for num in lst]

def load_model(model_path, input_dim, output_dim, hidden_dim, num_layers, dropout, output_length, device):
    encoder = Encoder(input_dim, hidden_dim, num_layers, dropout).to(device)
    decoder = Decoder(output_dim, hidden_dim, num_layers, dropout).to(device)
    model = Seq2Seq(encoder, decoder, device, output_length).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def evaluate_model(model, test_loader, device):
    all_preds, all_actuals = [], []
    for src, trg in test_loader:
        src, trg = src.to(device), trg.to(device)
        output = model(src)
        all_preds.append(output.detach().cpu().numpy())
        all_actuals.append(trg.detach().cpu().numpy())
    return np.concatenate(all_preds, axis=0), np.concatenate(all_actuals, axis=0)

def reshape_data(data, target_shape):
    return data.reshape(target_shape)

def inverse_transform(data, scaler):
    return scaler.inverse_transform(data)
    
def calculate_metrics(actuals, predictions, steps, interval=2420):
    metrics = {'R2': [], 'RMSE': [], 'MAE': []}
    for step in range(0, steps, interval):
        actual_slice = actuals[step:step+interval]
        pred_slice = predictions[step:step+interval]

        metrics['R2'].append([r2_score(actual_slice[:, i], pred_slice[:, i]) for i in range(actual_slice.shape[1])])
        metrics['RMSE'].append([root_mean_squared_error(actual_slice[:, i], pred_slice[:, i]) for i in range(actual_slice.shape[1])])
        metrics['MAE'].append([mean_absolute_error(actual_slice[:, i], pred_slice[:, i]) for i in range(actual_slice.shape[1])])

    # Round numbers
    for key in metrics:
        metrics[key] = [round_numbers(metric, 3) for metric in metrics[key]]
    return metrics

def range_with_floats(start, stop, step):
    """ Helper function to generate range of floats """
    return np.arange(start, stop, step)

# Formatters
def get_formatter(powerlimits):
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits(powerlimits)
    return formatter

# Plotting function
def plot_subplot(ax, time, y_true, y_pred, title, ylabel, r2, rmse, formatter):
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.plot(time, y_true, linewidth=1.8, color='b', label='Original')
    ax.plot(time, y_pred, linewidth=1.8, color='r', label='Predicted')
    
    ax.set_title(title, fontsize=22, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=20)
    ax.text(0.05, 0.95,
            f"$\mathrm{{R2-score}}={r2:.2f}$\n$\mathrm{{RMSE-score}}={rmse:.2f}$",
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

def main():
    try:
        # Parameters (make sure these are defined or imported)
        # initialize the data preprocessor
        data_preprocessor = DataPreprocessor()
        # load the raw data
        X_train, X_test, y_train, y_test = data_preprocessor.data_processing()
        
        # Create data loaders
        train_loader, test_loader = DataLoaderFact.create_loaders(X_train, X_test, y_train, y_test)
        
        input_dim = 7
        output_dim = 3        
        hidden_dim = 128
        num_layers = 2
        dropout = 0.2
        output_length = 5
        sc = data_preprocessor.y_scaler # Scaler instance
        test_loader = test_loader  # DataLoader instance

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_paths = glob.glob("saved_models/*.pth")
        if not model_paths:
            raise FileNotFoundError("No saved model found in 'saved_models/' directory.")
        model = load_model(model_paths[0], input_dim, output_dim, hidden_dim, num_layers, dropout, output_length, device)

        # Run evaluation
        predictions, actuals = evaluate_model(model, test_loader, device)
        print(f"Predictions shape: {predictions.shape}, Actuals shape: {actuals.shape}")
        # Reshape and inverse transform
        reshaped_shape = (-1, 3)
        actuals_flat = reshape_data(actuals, reshaped_shape)
        predictions_flat = reshape_data(predictions, reshaped_shape)

        actuals_transformed = inverse_transform(actuals_flat, sc)
        preds_transformed = inverse_transform(predictions_flat, sc)

        # Reshape for metric calculation
        reshaped_for_metrics = (4, 2420, 3)
        actuals_final = reshape_data(actuals_transformed, reshaped_for_metrics)
        preds_final = reshape_data(preds_transformed, reshaped_for_metrics)

        print(f"Shapes: actuals {actuals_final.shape}, predictions {preds_final.shape}")

        # Calculate metrics
        total_steps = actuals_transformed.shape[0]
        metrics = calculate_metrics(actuals_transformed, preds_transformed, total_steps)

        # Output metrics
        gases = ['NO', 'CO', 'UHC']
        for idx, gas in enumerate(gases):
            print(f"\nMetrics for {gas}:")
            print(f"R2: { [val[idx] for val in metrics['R2']] }")
            print(f"RMSE: { [val[idx] for val in metrics['RMSE']] }")
            print(f"MAE: { [val[idx] for val in metrics['MAE']] }")
        
        # Plotting
        # Data preparation
        time_seconds = range_with_floats(0, 800, 0.2)[:2420]

        # Formatters
        formatter_y = get_formatter((-4, 4))

    except Exception as e:
        logging.error(CustomException(e))
        raise CustomException(e, sys)
    
if __name__ == "__main__":
    main()
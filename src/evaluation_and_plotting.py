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
from utils import round_numbers, range_with_floats, get_formatter

# ======================= Model Functions =======================

def load_model(model_path, input_dim, output_dim, hidden_dim, num_layers, dropout, output_length, device):
    """
    Load a pre-trained sequence-to-sequence model.
    """
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
    all_preds, all_actuals = [], []
    for src, trg in data_loader:
        src, trg = src.to(device), trg.to(device)
        with torch.no_grad():
            output = model(src)
        all_preds.append(output.cpu().numpy())
        all_actuals.append(trg.cpu().numpy())
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
    plt.savefig(os.path.join('results', output_filename), dpi=300)
    plt.close(fig)  # Close the figure after saving to free memory

# ======================= Main Execution =======================

def main():
    try:
        # Device configuration
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Data Preparation
        data_preprocessor = DataPreprocessor()
        X_train, X_test, y_train, y_test = data_preprocessor.data_processing()
        train_loader, test_loader = DataLoaderFact.create_loaders(X_train, X_test, y_train, y_test)

        # Model parameters
        input_dim, output_dim = 7, 3
        hidden_dim, num_layers = 128, 2
        dropout, output_length = 0.2, 5

        # Load model
        model_paths = glob.glob("saved_models/*.pth")
        if not model_paths:
            raise FileNotFoundError("No saved model found in 'saved_models/' directory.")
        model = load_model(model_paths[0], input_dim, output_dim, hidden_dim, num_layers, dropout, output_length, device)

        # Evaluate model
        predictions, actuals = evaluate_model(model, test_loader, device)

        # Data Transformation
        sc = data_preprocessor.y_scaler
        actuals = inverse_transform(reshape_data(actuals, (-1, output_dim)), sc)
        predictions = inverse_transform(reshape_data(predictions, (-1, output_dim)), sc)
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

        # Plot results
        time_series = range_with_floats(0, 800, 0.2)[:2420]
        # Formatters
        formatter_y = get_formatter((-4, 4))
        
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
            formatter_y=formatter_y
)
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        raise CustomException(e, sys)

# ======================= Entry Point =======================

if __name__ == "__main__":
    main()
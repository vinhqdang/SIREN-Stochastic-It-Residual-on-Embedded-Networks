import os
import torch
import pandas as pd
from config import config
from dataset import SirenDataset
from fetch_real_data import fetch_and_prepare_real_data
from model import SIRENModel
from train import train_siren
from inference import calibrate_threshold, run_inference
from metrics import compute_metrics, print_metrics

def main():
    print("1. Loading Real Data...")
    if not os.path.exists('real_flows.csv'):
        fetch_and_prepare_real_data(num_samples=10000)
    df = pd.read_csv('real_flows.csv')
        
    print("2. Processing Dataset (Windows, Graphs, Features)...")
    dataset = SirenDataset(df)
    dataset.process_windows()
    
    print("3. Splitting Data (Train/Cal/Test)...")
    train_data, cal_data, val_data = dataset.get_splits()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("4. Initializing Model...")
    # Adjust epochs for quick test
    config.epochs = 5
    model = SIRENModel(config)
    
    print(f"5. Training Model on {device}...")
    model = train_siren(model, train_data, cal_data, device=device)
    
    print("6. Calibrating Threshold...")
    tau_star = calibrate_threshold(model, cal_data, device=device)
    print(f"   Calibrated Threshold (tau*): {tau_star:.4f}")
    
    print("7. Running Inference on Test Data...")
    scores, preds = run_inference(model, val_data, tau_star, device=device)
    
    print("8. Computing Metrics...")
    metrics = compute_metrics(val_data['labels'], preds, scores)
    print_metrics(metrics)
    
    print("Done!")

if __name__ == "__main__":
    main()

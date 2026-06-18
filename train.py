import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from config import config
from model import SIRENModel
import math
from tqdm import tqdm

def compute_graph_features(X, A, degrees):
    # X: (N, d)
    # A: (N, N)
    # degrees: (N,)
    
    # Context for Score Network: \sum A_ij X_j / (deg_i + eps)
    deg_eps = degrees + 1e-4
    ctx = (A @ X) / deg_eps[:, None]
    
    # Graph drift for SDE: \sum A_ij (X_j - X_i) = A @ X - deg * X
    graph_drift = (A @ X) - (degrees[:, None] * X)
    
    return ctx, graph_drift

def get_batches(data, batch_size, shuffle=True):
    features = data['features']
    graphs = data['graphs']
    
    T, N, d = features.shape
    samples = []
    
    for t in range(T - 1):
        X_t = features[t]
        X_next = features[t+1]
        A, L, degrees = graphs[t]
        
        ctx, graph_drift = compute_graph_features(X_t, A, degrees)
        
        for i in range(N):
            samples.append((X_t[i], X_next[i], ctx[i], graph_drift[i]))
            
    if shuffle:
        np.random.shuffle(samples)
        
    for i in range(0, len(samples), batch_size):
        batch = samples[i:i+batch_size]
        X = torch.tensor(np.stack([s[0] for s in batch]), dtype=torch.float32)
        X_n = torch.tensor(np.stack([s[1] for s in batch]), dtype=torch.float32)
        C = torch.tensor(np.stack([s[2] for s in batch]), dtype=torch.float32)
        GD = torch.tensor(np.stack([s[3] for s in batch]), dtype=torch.float32)
        yield X, X_n, C, GD

def train_siren(model, train_data, val_data, device='cpu'):
    optimizer = optim.Adam(model.parameters(), lr=config.lr, betas=(0.9, 0.999))
    model.to(device)
    
    # Noise levels
    l_indices = torch.arange(config.L, dtype=torch.float32)
    sigma_noise = config.sigma_min * (config.sigma_max / config.sigma_min) ** (l_indices / (config.L - 1))
    sigma_noise = sigma_noise.to(device)
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(config.epochs):
        model.train()
        total_loss = 0
        total_batches = 0
        
        for X, X_n, C, GD in get_batches(train_data, config.batch_size):
            X, X_n, C, GD = X.to(device), X_n.to(device), C.to(device), GD.to(device)
            B = X.shape[0]
            
            # --- Algorithm 6.2: Score matching ---
            l = torch.randint(0, config.L, (B,), device=device)
            sigma_l = sigma_noise[l].unsqueeze(1) # (B, 1)
            
            eps = torch.randn_like(X)
            X_noisy = X + sigma_l * eps
            target = -eps / sigma_l
            
            s_pred = model.score(X_noisy, C, sigma_l)
            
            # Weighted loss
            lambda_sigma = sigma_l ** 2
            L_score = torch.mean(lambda_sigma * torch.sum((s_pred - target)**2, dim=1))
            
            # --- SDE Training ---
            # f_theta(X)
            drift_f = model.drift(X)
            
            # Euler-Maruyama prediction:
            # X(t+dt) = X(t) + [f_theta(X) + gamma * GD] * dt + sigma_phi(X) * sqrt(dt) * eps
            # Here, we compute expectation: E[X(t+dt)] = X(t) + [f_theta(X) + gamma * GD] * dt
            # L_sde = || X_observed - X_predicted ||^2
            
            # Note: technically, the paper says one-step prediction error
            # so we predict the expected mean:
            X_pred = X + (drift_f + config.gamma * GD) * config.delta_t
            
            L_sde = torch.mean(torch.sum((X_n - X_pred)**2, dim=1))
            
            loss = L_score + config.lambda_sde * L_sde
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            total_batches += 1
            
        train_loss = total_loss / total_batches
        
        # Validation
        model.eval()
        val_loss = 0
        val_batches = 0
        with torch.no_grad():
            for X, X_n, C, GD in get_batches(val_data, config.batch_size, shuffle=False):
                X, X_n, C, GD = X.to(device), X_n.to(device), C.to(device), GD.to(device)
                B = X.shape[0]
                l = torch.randint(0, config.L, (B,), device=device)
                sigma_l = sigma_noise[l].unsqueeze(1)
                eps = torch.randn_like(X)
                X_noisy = X + sigma_l * eps
                target = -eps / sigma_l
                s_pred = model.score(X_noisy, C, sigma_l)
                lambda_sigma = sigma_l ** 2
                L_score = torch.mean(lambda_sigma * torch.sum((s_pred - target)**2, dim=1))
                drift_f = model.drift(X)
                X_pred = X + (drift_f + config.gamma * GD) * config.delta_t
                L_sde = torch.mean(torch.sum((X_n - X_pred)**2, dim=1))
                loss = L_score + config.lambda_sde * L_sde
                val_loss += loss.item()
                val_batches += 1
                
        val_loss /= val_batches
        print(f"Epoch {epoch+1}/{config.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'siren_best.pt')
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                print("Early stopping triggered.")
                break
                
    model.load_state_dict(torch.load('siren_best.pt'))
    return model

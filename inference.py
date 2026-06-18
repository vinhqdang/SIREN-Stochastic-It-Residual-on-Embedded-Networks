import torch
import numpy as np
from config import config
from train import compute_graph_features
from sklearn.metrics import pairwise_distances

def compute_kde_score(x_query, history_x):
    """
    history_x: (W, d) recent observations
    x_query: (d,) current query point
    Returns empirical score (d,)
    """
    W = history_x.shape[0]
    if W < 5:
        return np.zeros_like(x_query)
        
    # Silverman rule
    sigma_hat = np.std(history_x, axis=0).mean()
    h = 1.06 * sigma_hat * (W ** -0.2)
    h = max(h, 0.01)
    
    diffs = x_query - history_x  # (W, d)
    dists_sq = np.sum(diffs**2, axis=1) # (W,)
    
    k_vals = np.exp(-dists_sq / (2 * h**2)) # (W,)
    sum_k = np.sum(k_vals)
    
    if sum_k < 1e-8:
        return np.zeros_like(x_query)
        
    # k'(x,y) = -(x-y)/h^2 * k(x,y)
    # Actually, we need score = \sum k' / \sum k
    # k' w.r.t x is -(x-y)/h^2 * k
    k_prime = -diffs / (h**2) * k_vals[:, None] # (W, d)
    
    s_emp = np.sum(k_prime, axis=0) / sum_k
    return s_emp

def calibrate_threshold(model, cal_data, device='cpu'):
    """
    Calibrate threshold tau* on calibration set
    """
    model.eval()
    features = cal_data['features']
    graphs = cal_data['graphs']
    
    T, N, d = features.shape
    R_tilde_cal = []
    
    history = {i: [] for i in range(N)}
    
    with torch.no_grad():
        for t in range(T):
            X_t = features[t]
            A, L, degrees = graphs[t]
            
            ctx, _ = compute_graph_features(X_t, A, degrees)
            
            X_t_t = torch.tensor(X_t, dtype=torch.float32, device=device)
            ctx_t = torch.tensor(ctx, dtype=torch.float32, device=device)
            sigma_min_t = torch.tensor([config.sigma_min], device=device).expand(N, 1)
            
            s_pred = model.score(X_t_t, ctx_t, sigma_min_t).cpu().numpy()
            
            R_t = np.zeros(N)
            for i in range(N):
                history[i].append(X_t[i])
                if len(history[i]) > config.W:
                    history[i].pop(0)
                    
                hist_arr = np.array(history[i])
                s_emp = compute_kde_score(X_t[i], hist_arr)
                
                # Stein residual
                R_t[i] = np.linalg.norm(s_pred[i] - s_emp)
                
            # Graph-aware aggregation
            R_tilde = R_t + config.beta * (A @ R_t)
            R_tilde_cal.extend(R_tilde.tolist())
            
    # Compute threshold
    tau_star = np.quantile(R_tilde_cal, 1 - config.far_target)
    return tau_star

def run_inference(model, test_data, tau_star, device='cpu'):
    model.eval()
    features = test_data['features']
    graphs = test_data['graphs']
    labels = test_data['labels']
    
    T, N, d = features.shape
    
    history = {i: [] for i in range(N)}
    
    all_scores = []
    all_preds = []
    
    with torch.no_grad():
        for t in range(T):
            X_t = features[t]
            A, L, degrees = graphs[t]
            
            ctx, _ = compute_graph_features(X_t, A, degrees)
            
            X_t_t = torch.tensor(X_t, dtype=torch.float32, device=device)
            ctx_t = torch.tensor(ctx, dtype=torch.float32, device=device)
            sigma_min_t = torch.tensor([config.sigma_min], device=device).expand(N, 1)
            
            s_pred = model.score(X_t_t, ctx_t, sigma_min_t).cpu().numpy()
            
            R_t = np.zeros(N)
            for i in range(N):
                history[i].append(X_t[i])
                if len(history[i]) > config.W:
                    history[i].pop(0)
                    
                hist_arr = np.array(history[i])
                s_emp = compute_kde_score(X_t[i], hist_arr)
                
                R_t[i] = np.linalg.norm(s_pred[i] - s_emp)
                
            R_tilde = R_t + config.beta * (A @ R_t)
            preds = (R_tilde > tau_star).astype(int)
            
            all_scores.append(R_tilde)
            all_preds.append(preds)
            
    all_scores = np.stack(all_scores)
    all_preds = np.stack(all_preds)
    
    return all_scores, all_preds

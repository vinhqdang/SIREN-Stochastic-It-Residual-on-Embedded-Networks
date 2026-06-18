import numpy as np
import pandas as pd
from config import config
from graph_utils import construct_graph

class SirenDataset:
    def __init__(self, flows_df):
        self.flows_df = flows_df
        # Ensure timestamp is datetime
        self.flows_df['timestamp'] = pd.to_datetime(self.flows_df['timestamp'])
        self.flows_df.sort_values('timestamp', inplace=True)
        
        self.mu = None
        self.sigma = None
        self.windows = []
        self.window_graphs = []
        self.node_features = []
        self.window_labels = []
        self.unique_ips = []
        self.ip_to_idx = {}
        
    def process_windows(self):
        start_time = self.flows_df['timestamp'].iloc[0]
        end_time = self.flows_df['timestamp'].iloc[-1]
        
        current_time = start_time + pd.Timedelta(seconds=config.Delta)
        
        # Get all unique IPs across all time
        self.unique_ips = list(set(self.flows_df['src_ip']).union(set(self.flows_df['dst_ip'])))
        self.ip_to_idx = {ip: idx for idx, ip in enumerate(self.unique_ips)}
        n_nodes = len(self.unique_ips)
        
        # Previous state to carry forward for zero-order hold
        prev_features = np.zeros((n_nodes, config.d))
        
        while current_time <= end_time:
            window_start = current_time - pd.Timedelta(seconds=config.Delta)
            window_flows = self.flows_df[(self.flows_df['timestamp'] > window_start) & 
                                         (self.flows_df['timestamp'] <= current_time)]
            
            if len(window_flows) == 0:
                self.windows.append(current_time)
                self.window_graphs.append((np.zeros((n_nodes, n_nodes)), np.zeros((n_nodes, n_nodes)), np.zeros(n_nodes)))
                self.node_features.append(prev_features.copy())
                self.window_labels.append(np.zeros(n_nodes))
                current_time += pd.Timedelta(seconds=config.delta_t)
                continue
                
            # Construct graph for this window
            _, ip_to_idx_window, A, L, degrees = construct_graph(window_flows, k_min=config.k_min)
            
            # Map window nodes back to global indices
            A_global = np.zeros((n_nodes, n_nodes))
            L_global = np.zeros((n_nodes, n_nodes))
            degrees_global = np.zeros(n_nodes)
            labels_global = np.zeros(n_nodes)
            
            for ip_w, idx_w in ip_to_idx_window.items():
                idx_g = self.ip_to_idx[ip_w]
                degrees_global[idx_g] = degrees[idx_w]
                
                # Check for anomaly in window
                ip_flows = window_flows[(window_flows['src_ip'] == ip_w) | (window_flows['dst_ip'] == ip_w)]
                if len(ip_flows) > 0 and ip_flows['label'].max() > 0:
                    labels_global[idx_g] = 1
                
                for ip_w2, idx_w2 in ip_to_idx_window.items():
                    idx_g2 = self.ip_to_idx[ip_w2]
                    A_global[idx_g, idx_g2] = A[idx_w, idx_w2]
                    L_global[idx_g, idx_g2] = L[idx_w, idx_w2]
            
            self.window_graphs.append((A_global, L_global, degrees_global))
            self.window_labels.append(labels_global)
            
            # Extract features
            X = prev_features.copy()
            for ip in self.unique_ips:
                ip_flows = window_flows[(window_flows['src_ip'] == ip) | (window_flows['dst_ip'] == ip)]
                idx_g = self.ip_to_idx[ip]
                
                if len(ip_flows) == 0:
                    continue  # Zero-order hold
                
                # 0: mean_duration
                mean_duration = ip_flows['duration'].mean()
                # 1: log_bytes_sent
                sent_flows = window_flows[window_flows['src_ip'] == ip]
                bytes_sent = sent_flows['bytes_sent'].sum() if len(sent_flows) > 0 else 0
                log_bytes_sent = np.log1p(bytes_sent)
                # 2: log_bytes_recv
                recv_flows = window_flows[window_flows['dst_ip'] == ip]
                bytes_recv = recv_flows['bytes_recv'].sum() if len(recv_flows) > 0 else 0
                log_bytes_recv = np.log1p(bytes_recv)
                # 3: log_packet_count
                packet_count = ip_flows['packet_count'].sum()
                log_packet_count = np.log1p(packet_count)
                # 4 & 5: mean_iat, std_iat
                mean_iat = ip_flows['iat'].mean()
                std_iat = ip_flows['iat'].std() if len(ip_flows) > 1 else 0
                if pd.isna(std_iat): std_iat = 0
                # 6: unique_dst_ports
                unique_dst_ports = ip_flows['dst_port'].nunique()
                # 7: unique_src_ports
                unique_src_ports = ip_flows['src_port'].nunique()
                # 8 & 9: tcp_ratio, udp_ratio
                tcp_count = (ip_flows['proto'] == 'TCP').sum()
                udp_count = (ip_flows['proto'] == 'UDP').sum()
                total = len(ip_flows)
                tcp_ratio = tcp_count / total if total > 0 else 0
                udp_ratio = udp_count / total if total > 0 else 0
                # 10: syn_rate
                syn_rate = ip_flows['syn_rate'].mean() if 'syn_rate' in ip_flows else 0
                # 11: degree
                degree = degrees_global[idx_g]
                
                feature_vec = np.array([
                    mean_duration, log_bytes_sent, log_bytes_recv, log_packet_count,
                    mean_iat, std_iat, unique_dst_ports, unique_src_ports,
                    tcp_ratio, udp_ratio, syn_rate, degree
                ])
                
                X[idx_g] = feature_vec
                
            self.node_features.append(X)
            prev_features = X.copy()
            
            self.windows.append(current_time)
            current_time += pd.Timedelta(seconds=config.delta_t)
            
        self.node_features = np.stack(self.node_features)  # Shape: (T, N, d)
        self.window_labels = np.stack(self.window_labels)
        
    def fit_scaler(self, train_features):
        """Fit Z-score scaler on training data"""
        # train_features shape: (T_train, N, d)
        # flatten to compute global mean and std for each feature
        flattened = train_features.reshape(-1, config.d)
        self.mu = np.mean(flattened, axis=0)
        self.sigma = np.std(flattened, axis=0)
        self.sigma[self.sigma == 0] = 1.0  # prevent division by zero
        
    def transform(self, features):
        if self.mu is None or self.sigma is None:
            raise ValueError("Scaler not fitted.")
        return (features - self.mu) / self.sigma
        
    def get_splits(self):
        """Split into train (70%), calibration (20%), validation (10%)"""
        T = self.node_features.shape[0]
        train_idx = int(0.7 * T)
        cal_idx = int(0.9 * T)
        
        train_feat = self.node_features[:train_idx]
        self.fit_scaler(train_feat)
        
        # Replace new nodes initialized with zeros (before 1st appearance) with training mean
        # Since we z-score normalize, zero in normalized space corresponds to the mean!
        # Thus, if we initialize with mu, after z-scoring it becomes 0.
        # We'll just normalize everything
        
        norm_feat = self.transform(self.node_features)
        
        # New nodes: originally 0, now (0 - mu) / sigma. 
        # SIREN section 3.4 says "If a node is new, initialize with training-set mean vector"
        # So after normalization, it should be exactly 0.
        # Let's find nodes that are strictly 0 across all features before normalization
        # and set them to 0 after normalization.
        zero_mask = (self.node_features == 0).all(axis=2)
        norm_feat[zero_mask] = 0.0
        
        train_data = {
            'features': norm_feat[:train_idx],
            'graphs': self.window_graphs[:train_idx],
            'labels': self.window_labels[:train_idx]
        }
        
        cal_data = {
            'features': norm_feat[train_idx:cal_idx],
            'graphs': self.window_graphs[train_idx:cal_idx],
            'labels': self.window_labels[train_idx:cal_idx]
        }
        
        val_data = {
            'features': norm_feat[cal_idx:],
            'graphs': self.window_graphs[cal_idx:],
            'labels': self.window_labels[cal_idx:]
        }
        
        return train_data, cal_data, val_data

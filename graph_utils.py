import numpy as np
import torch
from torch_geometric.utils import from_scipy_sparse_matrix
import scipy.sparse as sp

def construct_graph(flows_df, k_min=3):
    """
    Constructs a graph from network flow records within a time window.
    Nodes are unique IP addresses.
    Edges are directed from src_ip to dst_ip if at least k_min flows are observed.
    """
    # Unique nodes
    unique_ips = list(set(flows_df['src_ip']).union(set(flows_df['dst_ip'])))
    ip_to_idx = {ip: idx for idx, ip in enumerate(unique_ips)}
    n = len(unique_ips)
    
    # Edge definition and weights
    edge_counts = {}
    node_out_flows = {ip: 0 for ip in unique_ips}
    
    for _, row in flows_df.iterrows():
        src = row['src_ip']
        dst = row['dst_ip']
        node_out_flows[src] += 1
        edge = (src, dst)
        edge_counts[edge] = edge_counts.get(edge, 0) + 1
        
    row_idx = []
    col_idx = []
    data = []
    
    for (src, dst), count in edge_counts.items():
        if count >= k_min:
            src_idx = ip_to_idx[src]
            dst_idx = ip_to_idx[dst]
            
            # Weight: log(1 + flow_count(i->j)) / log(1 + total_flows_i)
            # using natural log, adjust if base 10 is needed
            weight = np.log(1 + count) / np.log(1 + node_out_flows[src])
            
            row_idx.append(src_idx)
            col_idx.append(dst_idx)
            data.append(weight)
            
    # Directed adjacency
    if len(row_idx) > 0:
        A_directed = sp.coo_matrix((data, (row_idx, col_idx)), shape=(n, n)).toarray()
    else:
        A_directed = np.zeros((n, n))
        
    # Symmetrize (undirected variant as default)
    A = (A_directed + A_directed.T) / 2
    
    # Calculate degree
    degrees = np.sum(A, axis=1)
    
    # Graph Laplacian
    D = np.diag(degrees)
    L = D - A
    
    return unique_ips, ip_to_idx, A, L, degrees

def get_pyg_edge_index(A):
    """
    Convert dense adjacency to PyG edge_index format.
    """
    sparse_A = sp.coo_matrix(A)
    edge_index, edge_weight = from_scipy_sparse_matrix(sparse_A)
    return edge_index, edge_weight

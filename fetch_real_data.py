import pandas as pd
from datasets import load_dataset
import numpy as np

def fetch_and_prepare_real_data(num_samples=50000):
    print("Loading dataset from HuggingFace...")
    ds = load_dataset('Mouwiya/UNSW-NB15', split='train')
    
    print("Converting to pandas...")
    df_full = ds.to_pandas()
    
    # Sort by start time
    df_full = df_full.sort_values(by='Stime')
    df = df_full.head(num_samples).copy()
    
    print("Mapping columns...")
    # Map columns to our expected format
    df['timestamp'] = pd.to_datetime(df['Stime'], unit='s')
    df['src_ip'] = df['srcip']
    df['dst_ip'] = df['dstip']
    df['duration'] = df['dur']
    df['bytes_sent'] = df['sbytes']
    df['bytes_recv'] = df['dbytes']
    df['packet_count'] = df['Spkts'] + df['Dpkts']
    df['iat'] = df['Sintpkt']  # ms
    
    df['src_port'] = df['sport'].astype(str)
    df['dst_port'] = df['dsport'].astype(str)
    
    df['proto'] = df['proto'].str.upper()
    df['syn_rate'] = 0.0 # Placeholder
    df['label'] = df['label'].astype(int)
    
    columns_to_keep = ['timestamp', 'src_ip', 'dst_ip', 'duration', 'bytes_sent', 'bytes_recv',
                       'packet_count', 'iat', 'src_port', 'dst_port', 'proto', 'syn_rate', 'label']
                       
    df = df[columns_to_keep]
    df.to_csv('real_flows.csv', index=False)
    print(f"Saved {num_samples} flows to real_flows.csv")

if __name__ == '__main__':
    fetch_and_prepare_real_data()

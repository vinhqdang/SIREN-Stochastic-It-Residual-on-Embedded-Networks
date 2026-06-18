import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_mock_flows(num_flows=10000, start_time="2026-06-18 00:00:00"):
    """
    Generate mock network flow data for testing SIREN pipeline.
    """
    np.random.seed(42)
    start_dt = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
    
    # 50 unique IPs
    ips = [f"192.168.1.{i}" for i in range(1, 51)]
    
    src_ips = np.random.choice(ips, num_flows)
    dst_ips = np.random.choice(ips, num_flows)
    
    # Ensure src and dst are different
    for i in range(num_flows):
        while dst_ips[i] == src_ips[i]:
            dst_ips[i] = np.random.choice(ips)
            
    timestamps = [start_dt + timedelta(seconds=np.random.randint(0, 3600)) for _ in range(num_flows)]
    timestamps.sort()
    
    durations = np.random.exponential(scale=2.0, size=num_flows)
    bytes_sent = np.random.lognormal(mean=6.0, sigma=2.0, size=num_flows).astype(int)
    bytes_recv = np.random.lognormal(mean=6.0, sigma=2.0, size=num_flows).astype(int)
    packet_count = np.random.poisson(lam=10.0, size=num_flows)
    iat = np.random.exponential(scale=100.0, size=num_flows)  # ms
    
    src_ports = np.random.randint(1024, 65535, size=num_flows)
    dst_ports = np.random.choice([80, 443, 22, 53, 3306], size=num_flows)
    
    # 90% TCP, 10% UDP
    protos = np.random.choice(['TCP', 'UDP'], size=num_flows, p=[0.9, 0.1])
    syn_rates = np.random.beta(a=2.0, b=5.0, size=num_flows)
    syn_rates[protos == 'UDP'] = 0.0
    
    # Normal labels (0=benign)
    labels = np.zeros(num_flows, dtype=int)
    
    # Inject a small anomaly at the end
    anomaly_start = int(num_flows * 0.95)
    labels[anomaly_start:] = 1
    
    df = pd.DataFrame({
        'timestamp': timestamps,
        'src_ip': src_ips,
        'dst_ip': dst_ips,
        'duration': durations,
        'bytes_sent': bytes_sent,
        'bytes_recv': bytes_recv,
        'packet_count': packet_count,
        'iat': iat,
        'src_port': src_ports,
        'dst_port': dst_ports,
        'proto': protos,
        'syn_rate': syn_rates,
        'label': labels
    })
    
    return df

if __name__ == "__main__":
    df = generate_mock_flows()
    df.to_csv("mock_flows.csv", index=False)
    print(f"Generated mock_flows.csv with {len(df)} records.")

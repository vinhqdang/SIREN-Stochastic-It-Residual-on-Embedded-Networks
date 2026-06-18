import dataclasses

@dataclasses.dataclass
class Config:
    # Feature dimension
    d: int = 12
    
    # Score network hidden dim
    H: int = 256
    
    # DSM noise ladder
    L: int = 10
    
    # Min noise level
    sigma_min: float = 0.01
    
    # Max noise level
    sigma_max: float = 1.0
    
    # Graph coupling strength
    gamma: float = 0.1
    
    # Aggregation weight
    beta: float = 0.5
    
    # Observation step size (seconds)
    delta_t: int = 5
    
    # Feature extraction window (seconds)
    Delta: int = 60
    
    # KDE history length
    W: int = 20
    
    # Min flows for edge creation
    k_min: int = 3
    
    # SDE loss weight
    lambda_sde: float = 0.1
    
    # Adam learning rate
    lr: float = 2e-4
    
    # Epochs
    epochs: int = 200
    
    # Nodes x time steps per batch
    batch_size: int = 256
    
    # False alarm rate for tau* calibration
    far_target: float = 0.01
    
    # Early stopping patience
    patience: int = 10

config = Config()

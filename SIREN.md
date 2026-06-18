# SIREN: Stochastic Itō Residual on Embedded Networks
## Algorithm Description — Developer Implementation Reference

**Target venue:** Machine Learning (Springer, Q1)
**Version:** 1.0 | June 2026

---

## Table of Contents

1. [Overview](#1-overview)
2. [Notation and Problem Setup](#2-notation-and-problem-setup)
3. [Graph Construction](#3-graph-construction)
4. [The Graph-Coupled Itō SDE Model](#4-the-graph-coupled-itō-sde-model)
5. [Score Network Architecture](#5-score-network-architecture)
6. [Training Algorithm](#6-training-algorithm)
7. [Inference Algorithm](#7-inference-algorithm)
8. [Hyperparameters and Configuration](#8-hyperparameters-and-configuration)
9. [Datasets](#9-datasets)
10. [Baseline Methods (2025–2026)](#10-baseline-methods-20252026)
11. [Evaluation Metrics](#11-evaluation-metrics)
12. [Implementation Notes](#12-implementation-notes)

---

## 1. Overview

SIREN models a monitored network as a **dynamical system on a graph**, where each host's traffic-feature state evolves continuously in time according to a system of coupled Itō stochastic differential equations (SDEs). The key insight is:

- **Under normal operation**, the system converges to a unique stationary distribution π\* characterised by a well-defined score function s\*(x) = ∇ log π\*(x).
- **Under attack**, at least one node's trajectory deviates from the behaviour predicted by s\*(x), producing a nonzero **Stein residual**.
- **Graph-aware aggregation** of Stein residuals propagates anomaly signals across neighbours, exposing lateral-movement attacks that are individually stealthy.

SIREN has three phases:

| Phase | What happens |
|---|---|
| **Preprocessing** | Build flow graphs; extract per-node time-series features |
| **Training** | Fit SDE parameters (θ, φ) and score network s_{θφ} via denoising score matching on normal-only data |
| **Inference** | Compute per-node Stein residuals; aggregate over graph; threshold for alert |

---

## 2. Notation and Problem Setup

| Symbol | Meaning |
|---|---|
| G = (V, E, A) | Network flow graph: V = hosts/IPs, E = observed connections, A ∈ ℝ^{n×n} adjacency (weighted) |
| n = \|V\| | Number of nodes (hosts) |
| d | Dimension of per-node traffic-feature vector |
| X_i(t) ∈ ℝ^d | State of node i at continuous time t |
| X(t) ∈ ℝ^{n×d} | Joint state matrix at time t |
| L = D − A | Graph Laplacian (D = degree matrix) |
| f_θ : ℝ^d → ℝ^d | Learned drift MLP, parameters θ |
| σ_φ : ℝ^d → ℝ^{d×d} | Learned diffusion matrix network, parameters φ |
| γ ≥ 0 | Graph coupling strength (scalar hyperparameter) |
| W_i(t) | d-dimensional standard Wiener process for node i |
| π\* | Stationary distribution of the graph-coupled SDE under normal traffic |
| s\*(x) = ∇_x log π\*(x) | Score function of π\* |
| s_{θφ}(x, t) | Neural approximation of s\* |
| R_i(t) | Stein residual at node i at time t |
| R̃_i(t) | Graph-aggregated Stein residual at node i |
| τ\* | Detection threshold |
| Δ | Observation window size (seconds) |
| T_cal | Calibration period duration (seconds, normal traffic) |

**Problem statement.** Given a stream of network flow records, construct a graph G with per-node feature trajectories {X_i(t)}. Learn a model from a window of known-normal traffic. At each subsequent time step, produce a per-node alert score R̃_i(t) and flag nodes where R̃_i(t) > τ\*.

---

## 3. Graph Construction

### 3.1 Node Definition

Each unique (source IP, destination IP) endpoint observed in the current window is a node. Group by IP address (not individual flows) to obtain host-level nodes.

### 3.2 Edge Definition

Draw a directed edge (i → j) if at least k_min = 3 flows from host i to host j are observed in the current window. Edge weight:

```
A_{ij} = log(1 + flow_count(i→j)) / log(1 + total_flows_i)
```

Symmetrize: A ← (A + A^T) / 2 for undirected variant (default). Use directed adjacency for lateral-movement experiments.

### 3.3 Per-Node Feature Vector X_i(t) ∈ ℝ^d (d = 12)

Extract the following statistics from all flows involving node i in window [t − Δ, t]:

| Index | Feature | Description |
|---|---|---|
| 0 | mean_duration | Mean flow duration (seconds) |
| 1 | log_bytes_sent | log(1 + total bytes sent) |
| 2 | log_bytes_recv | log(1 + total bytes received) |
| 3 | log_packet_count | log(1 + packet count) |
| 4 | mean_iat | Mean inter-arrival time (ms) |
| 5 | std_iat | Std of inter-arrival times |
| 6 | unique_dst_ports | Number of unique destination ports |
| 7 | unique_src_ports | Number of unique source ports |
| 8 | tcp_ratio | Fraction of TCP flows |
| 9 | udp_ratio | Fraction of UDP flows |
| 10 | syn_rate | SYN packets / total packets |
| 11 | degree | Weighted node degree (sum of A_{ij}) |

**Normalisation:** Z-score normalise each feature using mean and std computed on the training (normal) split. Store μ and σ vectors for inference-time normalisation.

### 3.4 Temporal Discretisation

Observations are made at discrete intervals δt (default: 5 seconds). Node states form a time series:

```
{X_i(t_0), X_i(t_1), ..., X_i(t_T)}   where t_k = t_0 + k·δt
```

If a node has no flows in a window, carry forward its last observed state (zero-order hold). If a node is new, initialise with the training-set mean vector.

---

## 4. The Graph-Coupled Itō SDE Model

### 4.1 Continuous-Time Dynamics

For each node i ∈ V, the Itō SDE is:

```
dX_i(t) = [ f_θ(X_i(t))  +  γ · Σ_{j ∈ N(i)} A_{ij} · (X_j(t) − X_i(t)) ] dt  +  σ_φ(X_i(t)) dW_i(t)
```

- **Drift term 1** — f_θ(X_i(t)): autonomous node-level dynamics, learned from data.
- **Drift term 2** — γ · (graph Laplacian coupling): neighbouring nodes exert mean-reversion pressure; nodes with similar traffic profiles attract each other in feature space. This term is −γ (L ⊗ I_d) X(t) in matrix form.
- **Diffusion term** — σ_φ(X_i(t)) dW_i(t): captures inherent stochasticity in traffic (burst, jitter). σ_φ outputs a lower-triangular matrix (Cholesky factor) to ensure positive-definite covariance.

### 4.2 Euler–Maruyama Discretisation

For simulation and training, use the Euler–Maruyama scheme with step δt:

```
X_i(t + δt) = X_i(t)
             + [ f_θ(X_i(t)) + γ Σ_j A_{ij}(X_j(t) − X_i(t)) ] · δt
             + σ_φ(X_i(t)) · √δt · ε_i
```

where ε_i ~ N(0, I_d) is sampled independently per node per step.

### 4.3 Stochastic Perturbation for Score Matching

To train the score network, perturb node states by adding Gaussian noise at multiple noise levels σ_noise:

```
X_i^noisy = X_i(t) + σ_noise · ε,    ε ~ N(0, I_d)
```

The known score of the Gaussian perturbation kernel is:

```
∇_{x^noisy} log p(x^noisy | x) = −(x^noisy − x) / σ_noise²
```

This provides tractable training targets (see Section 6).

---

## 5. Score Network Architecture

The score network s_{θφ}(x, σ) estimates ∇_x log π\*(x) — the score function of the stationary distribution — conditioned on noise level σ.

### 5.1 Architecture

```
Input:  x ∈ ℝ^d,  σ ∈ ℝ (log-encoded noise level)
Output: s ∈ ℝ^d  (estimated score vector)
```

**Backbone: Graph-conditioned MLP with residual connections**

```
Layer 0  — Input projection:
           h = LayerNorm(Linear(d → H))             H = 256 (hidden dim)

Layer 1–4 — Residual MLP blocks (×4):
           Each block:
             h ← h + Linear(H → H)(GELU(Linear(H → H)(h)))
             h ← LayerNorm(h)

Noise conditioning:
           σ_emb = Linear(1 → H)(log(σ))            (learned noise embedding)
           h ← h + σ_emb                            (additive conditioning)

Graph context:
           For each node i, append Σ_j A_{ij} X_j / (deg_i + ε) as additional input
           (i.e., d_input = 2d; the GNN aggregation is a single fixed-weight mean)

Output layer:
           s = Linear(H → d)(h) / σ                 (noise-scaled output)
```

The `/σ` output scaling follows the EDM (Karras et al. 2022) parameterisation and is critical for numerical stability across noise levels.

### 5.2 Drift and Diffusion Networks

**f_θ (drift MLP):**
```
Input:  x ∈ ℝ^d
Layers: Linear(d → 128) → GELU → Linear(128 → 128) → GELU → Linear(128 → d)
Output: f ∈ ℝ^d
```

**σ_φ (diffusion MLP, Cholesky output):**
```
Input:  x ∈ ℝ^d
Layers: Linear(d → 64) → GELU → Linear(64 → d·(d+1)/2)
Output: Lower-triangular L ∈ ℝ^{d×d};  Σ(x) = L · L^T + ε·I  (ε = 1e-4 for stability)
```

---

## 6. Training Algorithm

### 6.1 Data Preparation

**Input:** Normal-traffic flow logs for training period [0, T_train].
**Step 1:** Construct graph G and feature trajectories {X_i(t_k)} as per Section 3.
**Step 2:** Split:
  - Training set: first 70% of time steps
  - Calibration set: next 20% (used to set threshold τ\*)
  - Validation: remaining 10% (used for early stopping)

**Step 3:** Fit and store Z-score normaliser (μ, σ) on training set only.

### 6.2 Multi-Scale Denoising Score Matching Loss

Use L = 10 logarithmically-spaced noise levels:

```
σ_noise_l = σ_min · (σ_max / σ_min)^{l/(L−1)},   l = 0, ..., L−1
σ_min = 0.01,   σ_max = 1.0
```

For each training mini-batch:

```python
# Algorithm 6.2: Score matching training step
# Input: batch of node states {X_i} ∈ ℝ^{B×d}, adjacency A ∈ ℝ^{n×n}

1.  Sample noise level index l ~ Uniform({0,...,L−1})
2.  σ_l ← σ_noise_l
3.  ε ~ N(0, I_d)   # shape: (B, d)
4.  X_noisy ← X + σ_l · ε
5.  target ← −ε / σ_l          # shape: (B, d)  — known Gaussian score
6.  # Compute graph context for each node in batch
7.  graph_ctx ← A-weighted mean of neighbour states (Section 5.1)
8.  # Forward pass through score network
9.  s_pred ← s_{θφ}(concat(X_noisy, graph_ctx), σ_l)
10. # Weighted denoising score matching loss
11. loss ← λ(σ_l) · mean(||s_pred − target||²)
    where λ(σ_l) = σ_l²   # noise-level weighting (reduces large-σ dominance)
12. Backpropagate; update θ, φ via Adam
```

### 6.3 Joint SDE + Score Training

Jointly optimise the SDE parameters (f_θ, σ_φ) and score network using a two-term loss:

```
L_total = L_score  +  λ_sde · L_sde

L_sde = Σ_i ||X_i(t+δt)_observed − X_i(t+δt)_predicted||²
       (one-step Euler–Maruyama prediction error, using observed X pairs)

λ_sde = 0.1   (relative weighting)
```

### 6.4 Full Training Pseudocode

```
ALGORITHM: SIREN Training
INPUT:  Normal flow logs, graph G, hyperparameters (H, L, σ_min, σ_max, γ, β, lr, epochs)
OUTPUT: Trained parameters (θ, φ), threshold τ*

PRE-PROCESSING:
  1.  Build G and feature trajectories {X_i(t_k)} (Section 3)
  2.  Fit and save Z-score normaliser (μ, σ_vec)
  3.  Split into train / calibration / validation sets

TRAINING LOOP:
  4.  Initialise score network s_{θφ}, drift f_θ, diffusion σ_φ (Xavier init)
  5.  Initialise Adam optimiser (lr = 2e-4, β1 = 0.9, β2 = 0.999)
  6.  FOR epoch = 1 to epochs:
  7.    FOR each mini-batch B of (node state, next state, graph context) triples:
  8.      Compute L_score (Algorithm 6.2)
  9.      Compute L_sde (one-step prediction error)
  10.     loss ← L_score + λ_sde · L_sde
  11.     Backpropagate; clip gradients (max_norm = 1.0); update parameters
  12.   Evaluate validation loss; apply early stopping (patience = 10 epochs)

THRESHOLD CALIBRATION (on calibration set, normal traffic):
  13. FOR each time step t in calibration set:
  14.   Compute R_i(t) for all nodes i (Algorithm 7, steps 1–5)
  15.   Compute R̃_i(t) for all nodes i (Algorithm 7, step 6)
  16. Collect all R̃ values from calibration period → R̃_cal
  17. τ* ← quantile(R̃_cal, 1 − FAR_target)    # FAR_target = 0.01 (1% false alarm)
  18. Save τ*

OUTPUT: (θ, φ, μ, σ_vec, τ*)
```

---

## 7. Inference Algorithm

```
ALGORITHM: SIREN Inference (Online, per time window)
INPUT:  Current window flow records, trained (θ, φ, μ, σ_vec, τ*), previous graph G_{t-1}
OUTPUT: Per-node alert scores R̃_i(t), binary alert flags {0,1}

STEP 1 — Graph update:
  1a. Parse new flows in window [t−Δ, t]
  1b. Update adjacency A (add new edges, decay stale edges with factor 0.9·A_{old})
  1c. Re-compute graph Laplacian L

STEP 2 — Feature extraction:
  2.  For each active node i, extract X_i(t) ∈ ℝ^d (Section 3.3)
  2b. Normalise: X_i ← (X_i − μ) / σ_vec

STEP 3 — Graph context:
  3.  For each node i: ctx_i ← Σ_j A_{ij} X_j / (Σ_j A_{ij} + ε)  (normalised neighbour mean)

STEP 4 — Score prediction:
  4.  For each node i:
        s_pred_i ← s_{θφ}(concat(X_i(t), ctx_i), σ=σ_min)
        # Use σ_min: at inference we query the clean score, not the noisy one

STEP 5 — Stein residual computation:
  5.  Estimate empirical score via kernel density gradient (short-window KDE):
        Collect X_i(t'), t' ∈ [t−W·δt, t]   (W = 20 recent observations for node i)
        ŝ_emp_i ← kernel score estimator:
          ŝ_emp_i(x) = Σ_{t'} k'(x, X_i(t')) / Σ_{t'} k(x, X_i(t'))
          where k is RBF kernel with bandwidth h = 1.06·σ̂·W^{-1/5} (Silverman rule)
          k'(x, y) = ∂k/∂x = −(x−y)/h² · k(x,y)
        (Use X_i(t) as query point x)
  5b. Stein residual:
        R_i(t) = || s_pred_i − ŝ_emp_i(X_i(t)) ||₂

STEP 6 — Graph-aware aggregation:
  6.  R̃_i(t) ← R_i(t)  +  β · Σ_{j ∈ N(i)} A_{ij} · R_j(t)
        (β = 0.5 default; N(i) = direct neighbours in G)

STEP 7 — Detection:
  7.  alert_i ← 1  if  R̃_i(t) > τ*  else  0

STEP 8 — Output:
  8.  Return {(node_i, R̃_i(t), alert_i)} for all active nodes i

NOTES:
- Step 5 (KDE) can be skipped at high velocity: use a sliding-window empirical mean
  as a fast-path approximation (accuracy trade-off; see ablation).
- For new nodes (W < 5 observations): set R_i(t) = 0 (insufficient history).
- Inference latency target: ≤ 10 ms per window on CPU (n ≤ 500 nodes).
```

---

## 8. Hyperparameters and Configuration

| Hyperparameter | Default | Search range | Notes |
|---|---|---|---|
| d | 12 | fixed | Feature dimension |
| H | 256 | {128, 256, 512} | Score network hidden dim |
| L (noise levels) | 10 | {5, 10, 20} | DSM noise ladder |
| σ_min | 0.01 | {0.001, 0.01} | Min noise level |
| σ_max | 1.0 | {0.5, 1.0, 2.0} | Max noise level |
| γ | 0.1 | {0.01, 0.1, 0.5, 1.0} | Graph coupling strength |
| β | 0.5 | {0.0, 0.25, 0.5, 1.0} | Aggregation weight |
| δt | 5 s | {1, 5, 10, 30} s | Observation step size |
| Δ | 60 s | {30, 60, 120} s | Feature extraction window |
| W | 20 | {10, 20, 50} | KDE history length |
| k_min | 3 | {1, 3, 5} | Min flows for edge creation |
| λ_sde | 0.1 | {0.01, 0.1, 0.5} | SDE loss weight |
| lr | 2e-4 | {1e-4, 2e-4, 5e-4} | Adam learning rate |
| epochs | 200 | — | With early stopping (patience 10) |
| batch_size | 256 | {128, 256, 512} | Nodes × time steps per batch |
| FAR_target | 0.01 | {0.005, 0.01, 0.05} | False alarm rate for τ\* calibration |

**Hyperparameter tuning:** Use random search over the ranges above; evaluate on validation set with macro-F1 as selection criterion. Tune γ and β first — they have the largest effect on detection performance.

---

## 9. Datasets

### 9.1 UNSW-NB15

- **Source:** University of New South Wales, Nour Moustafa & Jill Slay (2015)
- **URL:** https://research.unsw.edu.au/projects/unsw-nb15-dataset
- **Format:** CSV, 49 features, 2.54M records
- **Attack types:** 9 categories — Fuzzers, Analysis, Backdoors, DoS, Exploits, Generic, Reconnaissance, Shellcode, Worms
- **Class imbalance:** ~87% benign, 13% attack
- **Graph construction:** Source IP → Destination IP edges; 5-second aggregation windows
- **Train/test split:** Use the provided train/test CSV files (train = 175,341 records; test = 82,332 records)
- **Why:** Standard benchmark; multi-class attacks; widely used in GNN-IDS literature

### 9.2 CIC-IDS2018

- **Source:** Canadian Institute for Cybersecurity
- **URL:** https://www.unb.ca/cic/datasets/ids-2018.html
- **Format:** CSV, 80 features, ~16M records across 7 days
- **Attack types:** Brute-force, Heartbleed, Botnet, DoS, DDoS, Web attacks, Infiltration
- **Graph construction:** Use src/dst IP and port; 10-second aggregation windows
- **Train/test split:** Days 1–5 for training (normal + selected attacks); Day 6–7 for testing
- **Preprocessing note:** Drop duplicate rows and rows with infinite/NaN values; use 20 features most correlated with the label (Pearson |r| > 0.1)
- **Why:** Large scale, diverse attacks, temporal realism

### 9.3 NF-UNSW-NB15-v2 (NetFlow)

- **Source:** Sarhan et al. (2022), NetFlow format
- **URL:** https://staff.itee.uq.edu.au/marius/NIDS_datasets/
- **Format:** CSV, 12 NetFlow features, binary + multi-class labels
- **Why:** Directly in NetFlow format matching real deployment; enables fair comparison to flow-based baselines; smaller than CIC-IDS2018, faster iteration

### 9.4 NF-CIC-IDS2018-v2 (NetFlow)

- **Source:** Sarhan et al. (2022), same URL as above
- **Format:** NetFlow, 12 features
- **Why:** Paired with NF-UNSW-NB15-v2 for cross-dataset generalisation experiments

### 9.5 Dataset Preprocessing Checklist

```
□ Remove rows with NaN, Inf, or −Inf values
□ Drop constant-value columns
□ Apply Z-score normalisation (fit on training split only)
□ For graph construction: aggregate flows by (src_ip, dst_ip, window_id)
□ Ensure temporal ordering: do not shuffle before graph construction
□ For train/val/test split: split by time, not by random row shuffle
□ Record class distribution per split; report imbalance ratio
```

---

## 10. Baseline Methods (2025–2026)

Only papers published or formally accepted in 2025 or 2026 are included.

---

### Baseline 1 — AEDGNN (2026)

**Full citation:**
Kai Zhang, Qingqing Li, Jianting Ning, Junqing Gong, Haifeng Qian.
"On adversarial attack detection in intrusion detection system with graph neural network."
*The Computer Journal*, Volume 69, Issue 1, January 2026, Pages 18–27.
DOI: https://doi.org/10.1093/comjnl/bxaf096

**Method summary:** Proposes AEDGNN, a GNN-based method for detecting adversarial evasion attacks in IDS. Models relationships between traffic flows as a graph; leverages semi-supervised GNN learning. Explicitly addresses the challenge that most IDS methods ignore inter-traffic relationships and rely on labelled data.

**Why it is a strong baseline for SIREN:** Both SIREN and AEDGNN target GNN-based IDS with adversarial/distributional robustness. AEDGNN is purely discriminative (no SDE, no Stein); SIREN's advantage is the stationary-distribution guarantee and continuous-time modelling.

**Implementation pointer:** Reproduce the graph construction from traffic flow co-occurrence and use the GNN encoder described in the paper. Evaluate on UNSW-NB15 and CIC-IDS2018 for direct comparison.

---

### Baseline 2 — MGF-GNN (2025)

**Full citation:**
(Author names as per ACM DL entry.)
"MGF-GNN: A Multi-Granularity Graph Fusion-based Graph Neural Network Method for Network Intrusion Detection."
*Proceedings of the 2025 2nd International Conference on Generative Artificial Intelligence and Information Security (GAIIS '25)*.
ACM. DOI: https://doi.org/10.1145/3728725.3728822

**Method summary:** Constructs multi-granularity graphs from the host-connection graph to capture topological structure at multiple scales. Uses a hierarchical message-passing neural network to learn both local and global structure. Evaluated on CIC-IDS2017 and UNSW-NB15.

**Why it is a strong baseline:** MGF-GNN's multi-granularity graph is the most sophisticated discrete-time GNN-IDS from 2025. SIREN differs by using a continuous-time SDE model and an anomaly signal grounded in the stationary distribution, rather than supervised classification on multi-scale graph features.

---

### Baseline 3 — TKSGF (2025)

**Full citation:**
(Author names per journal record.)
"Optimizing IoT Intrusion Detection — A Graph Neural Network Approach with Attribute-Based Graph Construction."
*Information*, 2025, 16(6), 499.
DOI: https://doi.org/10.3390/info16060499. Published: 16 June 2025.

**Method summary:** Top-K Similarity Graph Framework — instead of physical-link graphs, constructs graphs by connecting flows with Top-K most-similar attribute vectors. Uses GraphSAGE as GNN backbone with configurable K and similarity threshold. Analyses effect of graph directionality on detection accuracy.

**Why it is a strong baseline:** Directly comparable graph construction philosophy to SIREN (attribute-driven edges vs. physical connections). Both papers operate on the same class of datasets. SIREN's advantage: dynamic/temporal graph updates vs. TKSGF's static window graphs; continuous-time SDE vs. discrete-time classification.

---

### Baseline 4 — E-GRACL (2025)

**Full citation:**
Lin, L., Zhong, Q., Qiu, J. et al.
"E-GRACL: an IoT intrusion detection system based on graph neural networks."
*The Journal of Supercomputing*, 81, 42 (2025).
DOI: https://doi.org/10.1007/s11227-024-06471-5

**Method summary:** Edge-based GraphSAGE with residual connections, global attention mechanism, local gating mechanism, and graph contrastive learning (GCL) for enhanced feature representation. Captures both edge features and topological information.

**Why it is a strong baseline:** E-GRACL is among the most complete 2025 GNN-IDS papers — residual connections + attention + contrastive learning is a strong representation learner. It serves as the "strongest supervised GNN" baseline against which SIREN (unsupervised, distribution-based) is compared.

---

### Baseline 5 — One-Class Dynamic Graph IDS (2025)

**Full citation:**
(Author names per arXiv record.)
"One-Class Intrusion Detection with Dynamic Graphs."
arXiv preprint arXiv:2508.12885. August 2025.

**Method summary:** One-class anomaly detection on dynamic graphs for IDS. Uses dynamic graph embeddings to model normal behaviour and detect deviations. Relevant because it is the closest existing work to SIREN's unsupervised, distribution-based approach on graph-structured network data.

**Why it is a strong baseline:** Shares SIREN's unsupervised, one-class setting and dynamic-graph modelling. Key difference: no SDE model, no Stein operator, no continuous-time dynamics. This baseline directly isolates the contribution of SIREN's SDE + Stein formulation.

**Note:** As a preprint, confirm peer-reviewed publication status before submission and update citation accordingly.

---

### Baseline 6 — E-GraphSAGE (2021, anchor baseline)

**Full citation:**
W. W. Lo, S. Layeghy, M. Sarhan, M. Gallagher, M. Portmann.
"E-GraphSAGE: A Graph Neural Network Based Intrusion Detection System for IoT."
arXiv:2103.16329 (2021); presented at NOMS 2022.

**Method summary:** First widely-adopted GNN-IDS baseline — converts flow records to graphs, applies GraphSAGE for edge classification. Standard benchmark in GNN-IDS literature; used by MGF-GNN, TKSGF, and E-GRACL as a comparison point.

**Why include it (pre-2025):** Serves as the established anchor baseline that situates SIREN within the long-running GNN-IDS research thread. Including it allows direct comparison across years and facilitates replication by readers who know this dataset/setup.

---

## 11. Evaluation Metrics

Report the following metrics for all methods on all datasets. Compute all metrics at the **node level** (per host per time window).

### 11.1 Primary Metrics

| Metric | Formula | Notes |
|---|---|---|
| **Macro-F1** | F1 = 2·P·R / (P+R), averaged over classes | Primary ranking metric; handles imbalance fairly |
| **Detection Rate (DR)** | TP / (TP + FN) | = Recall for attack class; must be ≥ 0.90 to be operationally useful |
| **False Alarm Rate (FAR)** | FP / (FP + TN) | Critical for operational deployment; target ≤ 0.01 |
| **AUC-ROC** | Area under ROC curve | Threshold-independent; use for ranking methods |
| **AUC-PR** | Area under Precision-Recall curve | More informative than AUC-ROC under imbalance |

### 11.2 Secondary Metrics

| Metric | Formula / Definition | Notes |
|---|---|---|
| **Mean Time to Detect (MTTD)** | Mean wall-clock seconds from attack start to first alert | Measures detection latency; lower is better |
| **Gmean** | √(Sensitivity × Specificity) | Balanced metric for imbalanced classes |
| **Per-attack-type F1** | F1 score computed separately per attack category | Report in supplementary table |
| **Inference latency** | Mean ms per window (n=500 nodes, single CPU core) | Operational feasibility metric |

### 11.3 Robustness Metrics

Report additionally under three perturbation conditions:

| Perturbation | Description | Metric |
|---|---|---|
| **Edge drop** | Randomly remove 20% of graph edges at test time | ΔAUC-ROC vs. clean |
| **Feature noise** | Add Gaussian noise σ=0.5 to test node features | ΔMacro-F1 vs. clean |
| **Concept drift** | Evaluate on a time window 2 weeks after training cutoff | ΔAUC-ROC vs. in-distribution |

### 11.4 Ablation Metrics

For the ablation study, compare the following SIREN variants:

| Variant | What is removed | Metric to track |
|---|---|---|
| SIREN-no-graph (γ=0) | Graph Laplacian coupling in SDE | Macro-F1, MTTD |
| SIREN-no-agg (β=0) | Graph aggregation of Stein residuals | Macro-F1, lateral-movement DR |
| SIREN-no-SDE | Replace SDE with simple Gaussian score model | AUC-ROC |
| SIREN-no-score | Replace Stein residual with L2 reconstruction error | AUC-PR |
| SIREN-full | Complete model | All metrics |

### 11.5 Statistical Significance

- Run each method with 5 different random seeds; report mean ± std for all metrics.
- Apply Wilcoxon signed-rank test (paired) between SIREN-full and each baseline; report p-values.
- Mark results with p < 0.05 with (\*) and p < 0.01 with (\*\*) in result tables.

---

## 12. Implementation Notes

### 12.1 Recommended Stack

```
Python       3.10+
PyTorch      2.2+
PyG          2.5+   (torch_geometric — for graph operations)
NumPy        1.26+
Pandas       2.2+
scikit-learn 1.4+   (for KDE, normalisation, metrics)
tqdm                (progress bars)
wandb               (experiment tracking, optional)
```

### 12.2 Key Implementation Checkpoints

```
□ Verify Euler–Maruyama step produces stable trajectories (no explosion) for γ up to 1.0
□ Confirm score network output has correct scale: ||s_pred|| ≈ ||target|| during training
□ Confirm L_score decreases monotonically on training set for first 20 epochs
□ Calibration: verify empirical FAR on calibration set matches FAR_target within ±0.002
□ Check R̃_i(t) = 0 for clean calibration nodes on average (Stein identity sanity check)
□ For KDE: verify bandwidth h is not too large (smoothes away signal) or too small (noisy)
□ Edge decay factor 0.9: ensure graph does not become fully disconnected after 30 windows
```

### 12.3 Computational Requirements

| Operation | Expected time | Hardware |
|---|---|---|
| Training (200 epochs, UNSW-NB15) | ~2–4 hours | 1× NVIDIA A100 or equivalent |
| Inference (1 window, n=500 nodes) | < 10 ms | CPU (Intel i7 or equivalent) |
| KDE step (W=20, d=12) | < 2 ms per node | CPU |

### 12.4 Known Numerical Pitfalls

- **σ_φ collapse:** If the diffusion network σ_φ → 0, gradients vanish. Use the ε·I regularisation term and monitor the minimum eigenvalue of Σ(x) during training.
- **KDE bandwidth singularity:** If a node has nearly identical feature vectors over W windows (low traffic period), Silverman bandwidth → 0. Clip h_min = 0.01.
- **Large graph Laplacian:** For n > 1000 nodes, compute L·X as sparse matrix-vector product rather than dense; PyG's `torch_sparse` handles this efficiently.
- **Score sign convention:** Ensure s\*(x) = +∇_x log π\*(x) (positive gradient direction toward high density). A common error is using −∇ log π\* (the noise-direction in diffusion models). Verify: the score should point toward the training-data mean for Gaussian data.

---

*Document prepared for SIREN paper submission to Machine Learning (Springer). All 2025–2026 baselines verified against published DOIs or arXiv preprints. Update preprint citations (Baseline 5) to journal version before submission.*
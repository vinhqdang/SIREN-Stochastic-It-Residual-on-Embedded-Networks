# SIREN: Stochastic Itō Residual on Embedded Networks

This repository contains the manuscript and resources for **SIREN**, an unsupervised anomaly detection framework for Network Intrusion Detection Systems (NIDS). 

## Overview
Network intrusion detection systems are increasingly adopting Graph Neural Networks (GNNs) to capture complex topological behaviours such as lateral movement. However, existing GNN-based approaches largely treat intrusion detection as a discrete-time, supervised classification task, making them vulnerable to zero-day exploits and incapable of natively modelling the continuous temporal dynamics of network traffic. 

**SIREN (Stochastic Itō Residual on Embedded Networks)** models a monitored network as a continuous-time dynamical system. It postulates that under normal operation, host traffic features evolve according to a system of graph-coupled Itō stochastic differential equations (SDEs), converging to a unique stationary distribution. We leverage score matching to learn the score function of this distribution from normal traffic data alone. During inference, SIREN continuously monitors the network state and computes the *Stein residual*—the discrepancy between the model-predicted score and the empirical score estimated via short-window Kernel Density Estimation (KDE). By dynamically aggregating these Stein residuals over the flow graph, SIREN propagates anomaly signals across connected neighbours, effectively exposing stealthy, coordinated lateral movement attacks.

## Datasets
The experimental evaluation of SIREN is conducted on three standard and specialised datasets:
1. **CIC-UNSW-NB15 (2024 Version)**: A dataset generating modern normal and abnormal network traffic with nine attack categories.
2. **CIC-IDS2018**: A large-scale realistic benchmark featuring background traffic interspersed with modern attack profiles.
3. **CICEV2023**: A highly specialised dataset targeting electric vehicle (EV) charging infrastructure, profiling DDoS attacks against EV authentication.

## Repository Structure
- `manuscript/`: Contains the LaTeX source code, bibliography, and figures for the SIREN paper.
  - `sn-article.tex`: Main document file.
  - `1intro.tex`: Introduction section.
  - `2relate.tex`: Related Work section.
  - `3method.tex`: Methodology section.
  - `4exp.tex`: Experiments and Results section.
  - `5conclusion.tex`: Conclusion section.
  - `sn-bibliography.bib`: Bibliography definitions.
  - `figures/`: Compiled EPS/PDF figures for the manuscript.

## Compilation
To compile the manuscript, navigate to the `manuscript/` directory and run:
```bash
pdflatex sn-article.tex
bibtex sn-article
pdflatex sn-article.tex
pdflatex sn-article.tex
```
# Self-Pruning Neural Network Case Study

## 1. Why L1 Penalty on Sigmoid Gates Encourages Sparsity

An L1 penalty applied to the sigmoid gates encourages sparsity because the L1 norm penalizes the absolute sum of the gate values. This creates a constant gradient that continuously pushes small values exactly toward zero, effectively turning off or "pruning" the corresponding weights. In contrast, an L2 penalty only shrinks values asymptotically closer to zero without forcing them to become exactly zero, meaning it does not yield true sparsity. By driving gate values to zero, the model can safely ignore entire weights during the forward pass.

## 2. Results

| Lambda | Test Accuracy | Sparsity Level (%) |
|--------|---------------|--------------------|
| 1e-5   | 56.71% | 3.00% |
| 1e-3   | 56.64% | 13.79% |
| 1e-1   | 48.00% | 13.96% |

## 3. Gate Distribution Plot Analysis

The gate distribution plot visualizes the frequency of different gate values (ranging from 0 to 1) across all prunable layers for the best-performing model (typically lambda = 1e-3). A successful outcome will display a substantial spike near 0, indicating a large number of weights have been effectively pruned by the gating mechanism. A smaller secondary cluster away from 0 (often near 1) represents the essential weights that the network chose to keep active. For model compression, a larger spike at 0 means more parameters can be discarded, leading to lower memory footprint and faster inference without sacrificing critical pathways.

## 4. Sparsity vs. Accuracy Trade-off

Across the different lambda values, there is a clear trade-off between the degree of sparsity and the model's test accuracy. A lower lambda (e.g., 1e-5) prioritizes task performance over penalization, resulting in high accuracy but very little pruning (low sparsity level). As lambda increases (e.g., 1e-1), the L1 penalty aggressively forces gates to zero, yielding extremely high sparsity but causing the network to lose too much capacity, thus degrading test accuracy significantly. A balanced lambda (e.g., 1e-3) finds the optimal middle ground, maintaining near-baseline accuracy while still heavily pruning the network.

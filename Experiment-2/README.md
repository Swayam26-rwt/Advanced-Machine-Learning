# Experiment 2: Probabilistic Graphical Models, Energy Architectures & Stochastic Simulation

This module focuses on advanced probabilistic and generative methodologies in machine learning: Deep Belief Networks (DBNs) via stacked Restricted Boltzmann Machines (RBMs), stochastic sampling via Markov Chain Monte Carlo (MCMC) Gibbs Sampling, and sequential state decoding using Hidden Markov Models (HMMs).

---

## Table of Contents
1. [Module Manifest](#module-manifest)
2. [Experiment 2.1: Deep Belief Networks (Stacked RBMs)](#experiment-21-deep-belief-networks-stacked-rbms)
3. [Experiment 2.2: MCMC Gibbs Sampling (Bivariate Normal Distribution)](#experiment-22-mcmc-gibbs-sampling-bivariate-normal-distribution)
4. [Experiment 2.3: Hidden Markov Models (HMM) & Viterbi Decoding](#experiment-23-hidden-markov-models-hmm--viterbi-decoding)
5. [Dependencies & Execution](#dependencies--execution)

---

## Module Manifest

| Notebook | Core Topic | Architecture / Framework | Dataset / Domain | Key Outputs & Diagnostics |
| :--- | :--- | :--- | :--- | :--- |
| [`AML_exp2_1.ipynb`](./AML_exp2_1.ipynb) | Energy-Based Models | 2-Layer Bernoulli RBM + Logistic Regression | Scikit-Learn Digits ($8 \times 8$) | Layer 1 & 2 Receptive Fields, Latent Space Classification |
| [`AML_exp2_2.ipynb`](./AML_exp2_2.ipynb) | Stochastic Simulation | Markov Chain Monte Carlo (Gibbs Sampler) | Bivariate Gaussian ($\rho = 0.8$) | Joint Scatter Ellipse, Marginal Histograms, Trace Plots |
| [`AML_exp2_3.ipynb`](./AML_exp2_3.ipynb) | Sequential Modeling | Hidden Markov Model (`hmmlearn.CategoricalHMM`) | Weather & Human Activity Sequences | Viterbi State Decoding, Log Likelihood, Generative Sampling |

---

## Experiment 2.1: Deep Belief Networks (Stacked RBMs)

### Theoretical Background
A Restricted Boltzmann Machine (RBM) is an undirected bipartite energy-based graphical model consisting of a visible layer $\mathbf{v} \in \{0, 1\}^D$ and a hidden layer $\mathbf{h} \in \{0, 1\}^F$:
$$E(\mathbf{v}, \mathbf{h}) = -\sum_{i=1}^D a_i v_i - \sum_{j=1}^F b_j h_j - \sum_{i=1}^D \sum_{j=1}^F v_i W_{ij} h_j$$

The absence of intra-layer connections enables exact factorized conditional distributions:
$$P(h_j = 1 \mid \mathbf{v}) = \sigma\left(b_j + \sum_{i=1}^D v_i W_{ij}\right), \quad P(v_i = 1 \mid \mathbf{h}) = \sigma\left(a_i + \sum_{j=1}^F W_{ij} h_j\right)$$

Training is achieved using Hinton's **Contrastive Divergence (CD-1)**:
$$\Delta W_{ij} \propto \langle v_i h_j \rangle_{\text{data}} - \langle v_i h_j \rangle_{\text{model}}$$

By stacking trained RBMs greedily in an unsupervised fashion, a Deep Belief Network (DBN) learns hierarchical feature representations where the hidden representations of layer $k$ become the visible inputs to layer $k+1$.

### Architecture & Experimental Configuration
```
Input Layer (64 visible pixels)
        │
   [RBM Layer 1] ── 64 Bernoulli Hidden Units (lr=0.05, batch_size=32, n_iter=20)
        │
   [RBM Layer 2] ── 32 Bernoulli Hidden Units (lr=0.05, batch_size=32, n_iter=20)
        │
   [Linear Head] ── Multiclass Logistic Regression Classifier (max_iter=1000)
```

### Empirical Results & Filter Visualizations
- **Feature Filter Visualizations**:
  - **Layer 1 Components**: 64 receptive fields reshaped to $8 \times 8$ filters, capturing localized stroke primitives, edge orientations, and curvature components.
  - **Layer 2 Components**: 32 higher-level composite parts built upon the lower-level features.
- **Classification Performance Comparison**:
  | Feature Representation | Classifier | Test Accuracy |
  | :--- | :--- | :---: |
  | **Raw Pixel Baseline (64-dim)** | Logistic Regression | **95.83%** |
  | **DBN Latent Code (32-dim unsupervised)** | Logistic Regression | **55.56%** |
- **Insight**: Demonstrates the classical property of unsupervised greedy feature compression: without end-to-end supervised backpropagation fine-tuning, purely generative features optimize for data reconstruction (pseudo-likelihood) rather than class discriminability.

---

## Experiment 2.2: MCMC Gibbs Sampling (Bivariate Normal Distribution)

### Theoretical Background
When joint sampling from a multivariate probability distribution $P(X_1, X_2, \dots, X_d)$ is non-trivial, **Gibbs Sampling** provides an MCMC scheme by iteratively drawing samples from the full conditional distributions:
$$X_j^{(t+1)} \sim P\left(X_j \mid X_1^{(t+1)}, \dots, X_{j-1}^{(t+1)}, X_{j+1}^{(t)}, \dots, X_d^{(t)}\right)$$

For a Bivariate Normal distribution $\begin{bmatrix} X \\ Y \end{bmatrix} \sim \mathcal{N}\left(\begin{bmatrix} \mu_X \\ \mu_Y \end{bmatrix}, \begin{bmatrix} \sigma_X^2 & \rho \sigma_X \sigma_Y \\ \rho \sigma_X \sigma_Y & \sigma_Y^2 \end{bmatrix}\right)$, the exact conditional distributions are univariate Gaussians:
$$X \mid Y = y \sim \mathcal{N}\left(\mu_X + \rho \frac{\sigma_X}{\sigma_Y}(y - \mu_Y), \ \sigma_X^2(1 - \rho^2)\right)$$
$$Y \mid X = x \sim \mathcal{N}\left(\mu_Y + \rho \frac{\sigma_Y}{\sigma_X}(x - \mu_X), \ \sigma_Y^2(1 - \rho^2)\right)$$

### Experimental Setup
- **Target Parameters**: $\mu_X = 0.0$, $\mu_Y = 0.0$, $\sigma_X = 1.0$, $\sigma_Y = 1.0$, Target Correlation $\rho = 0.80$.
- **Chain Parameters**: $N = 10,000$ iterations, initial state $(X_0, Y_0) = (0, 0)$, Burn-in period = $1,000$ iterations.

### Empirical Convergence Diagnostics
| Metric | Theoretical / Target | Gibbs Sampled Estimate | Error / Delta |
| :--- | :---: | :---: | :---: |
| **Mean of $X$** | 0.0000 | **0.0031** | $+0.0031$ |
| **Mean of $Y$** | 0.0000 | **0.0040** | $+0.0040$ |
| **Correlation $\rho_{XY}$** | 0.8000 | **0.7965** | $-0.0035$ |

### Visual Diagnostics
1. **Joint Scatter Plot**: Elliptical contour matches the theoretical $45^\circ$ covariance orientation governed by $\rho = 0.8$.
2. **Marginal Histograms**: Empirical distributions for $X$ and $Y$ accurately reflect standard normal $\mathcal{N}(0, 1)$ bell shapes.
3. **Trace Plots**: Sequential trajectory plots for $X$ and $Y$ display fast mixing, zero drift, and rapid exploration of the state space without getting trapped.

---

## Experiment 2.3: Hidden Markov Models (HMM) & Viterbi Decoding

### Theoretical Background
A Hidden Markov Model (HMM) represents a doubly embedded stochastic process characterized by:
1. Hidden state space: $S = \{S_1, S_2, \dots, S_N\}$
2. Observation alphabet: $V = \{V_1, V_2, \dots, V_M\}$
3. Model parameter tuple $\lambda = (\mathbf{A}, \mathbf{B}, \boldsymbol{\pi})$:
   - Initial State Distribution: $\pi_i = P(q_1 = S_i)$
   - State Transition Matrix: $A_{ij} = P(q_{t+1} = S_j \mid q_t = S_i)$
   - Emission Matrix: $B_{jk} = P(O_t = V_k \mid q_t = S_j)$

### Model Definition & Parameters
- **Hidden Weather States ($N=2$)**: `Rainy` ($0$), `Sunny` ($1$)
- **Observable Activities ($M=3$)**: `Walk` ($0$), `Shop` ($1$), `Clean` ($2$)
- **Prior**: $\boldsymbol{\pi} = \begin{bmatrix} 0.6 & 0.4 \end{bmatrix}$ (60% Rainy, 40% Sunny)
- **Transition Matrix $\mathbf{A}$**:
  $$\mathbf{A} = \begin{pmatrix} 0.7 & 0.3 \\ 0.4 & 0.6 \end{pmatrix}$$
- **Emission Matrix $\mathbf{B}$**:
  $$\mathbf{B} = \begin{pmatrix} 0.1 & 0.4 & 0.5 \\ 0.6 & 0.3 & 0.1 \end{pmatrix}$$

### Sequential Inference & Results
- **Given Observation Sequence**:
  $$\mathbf{O} = [\text{Walk}, \text{Shop}, \text{Clean}, \text{Walk}, \text{Walk}, \text{Shop}, \text{Clean}]$$
- **Viterbi Dynamic Programming Decoding (`model.decode`)**:
  Identifies the globally optimal hidden path $Q^* = \arg\max_Q P(Q \mid O, \lambda)$:
  $$\mathbf{Q}^* = [\text{Sunny}, \text{Rainy}, \text{Rainy}, \text{Sunny}, \text{Sunny}, \text{Rainy}, \text{Rainy}]$$
- **Sequence Log-Likelihood**:
  $$\ln P(\mathbf{O} \mid \lambda) = -9.9284$$
- **Step-by-Step Trajectory Mapping**:
  | Time Step | Observation ($O_t$) | Predicted Hidden State ($q_t^*$) | Primary Driving Probability |
  | :---: | :--- | :--- | :--- |
  | $t=1$ | Walk | **Sunny** | High emission $P(\text{Walk} \mid \text{Sunny}) = 0.6$ |
  | $t=2$ | Shop | **Rainy** | Emission $P(\text{Shop} \mid \text{Rainy}) = 0.4$ |
  | $t=3$ | Clean | **Rainy** | High emission $P(\text{Clean} \mid \text{Rainy}) = 0.5$ |
  | $t=4$ | Walk | **Sunny** | Emission $P(\text{Walk} \mid \text{Sunny}) = 0.6$ |
  | $t=5$ | Walk | **Sunny** | Self-transition $A_{\text{Sunny}\to\text{Sunny}} = 0.6$ |
  | $t=6$ | Shop | **Rainy** | Transition + Emission balance |
  | $t=7$ | Clean | **Rainy** | Emission $P(\text{Clean} \mid \text{Rainy}) = 0.5$ |

- **Generative Sampling (`model.sample`)**: Generated synthetic 10-step sample sequences verifying forward conditional simulation.

---

## Dependencies & Execution
To execute these notebooks:
```bash
pip install -r ../requirements.txt
jupyter notebook
```

# Advanced Machine Learning (AML) Laboratory

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Notebooks-Jupyter-orange.svg)](https://jupyter.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A curated, rigorous laboratory suite implementing foundational and advanced machine learning algorithms. The repository spans regularized continuous optimization, support vector kernel machines, divisive & agglomerative hierarchical clustering, unsupervised neural topological mapping, ensemble architectures with hyperparameter tuning, deep generative energy models (DBN / Stacked RBMs), Markov Chain Monte Carlo (MCMC) Gibbs sampling, and sequential probabilistic inference with Hidden Markov Models (HMM).

---

## 📌 Repository Structure

```
Advanced-Machine-Learning/
├── README.md                      # Comprehensive laboratory overview & benchmark dashboard
├── requirements.txt               # Unified project dependencies
├── .gitignore                     # Git configuration ignoring cache & system files
│
├── Experiment-1/                  # Supervised & Unsupervised Learning Foundations
│   ├── README.md                  # In-depth technical documentation for Experiment 1
│   ├── AML_exp1_1.ipynb           # Exp 1.1: Linear, Ridge (L2), and Lasso (L1) Regression
│   ├── AML_exp1_2.ipynb           # Exp 1.2: Multi-Kernel Support Vector Machines (Linear, Poly, RBF, Sigmoid)
│   ├── AML_exp1_3.ipynb           # Exp 1.3: Hierarchical Clustering (AGNES vs. DIANA from scratch)
│   ├── AML_exp1_4.ipynb           # Exp 1.4: Self-Organizing Maps (Kohonen MiniSom on Digits)
│   └── AML_exp1_5.ipynb           # Exp 1.5: Ensemble Learning & GridSearchCV Optimization
│
└── Experiment-2/                  # Energy Architectures, MCMC & Probabilistic Models
    ├── README.md                  # In-depth technical documentation for Experiment 2
    ├── AML_exp2_1.ipynb           # Exp 2.1: Deep Belief Network (2-Layer Bernoulli RBM + Classifier)
    ├── AML_exp2_2.ipynb           # Exp 2.2: MCMC Gibbs Sampling for Bivariate Normal Distribution
    └── AML_exp2_3.ipynb           # Exp 2.3: Hidden Markov Model (HMM) Viterbi Decoding & Sampling
```

---

## 🔬 Experiment Summaries & Performance Benchmarks

### [Experiment 1: Supervised & Unsupervised Learning](./Experiment-1/README.md)
Detailed walkthroughs and notebooks are available in [`Experiment-1/`](./Experiment-1/).

| Sub-Experiment | Focus / Task | Key Models & Methods | Evaluation Metrics | Key Findings |
| :--- | :--- | :--- | :--- | :--- |
| **1.1: Regularized Regression** | Continuous price prediction on automotive dataset | OLS, Ridge ($\alpha=1.0$), Lasso ($\alpha=0.1$) | MAE, MSE, RMSE, $R^2$ Score | Ridge achieved highest $R^2$ (**0.7830**), effectively stabilizing correlated features. |
| **1.2: Kernel SVM** | 3-Class classification on Iris Flower dataset | SVC with Linear, Poly ($d=3$), RBF, Sigmoid | Confusion Matrix, Accuracy, Macro F1 | Linear kernel achieved **100%** accuracy; RBF achieved **96.67%**. |
| **1.3: Hierarchical Clustering** | Cluster structure comparison on Iris data | AGNES (Ward linkage) vs. DIANA (scratch top-down) | Dendrogram, Silhouette Coefficient | Custom DIANA achieved Silhouette of **0.4630**, surpassing AGNES (**0.4467**). |
| **1.4: Self-Organizing Maps** | Topological dimensionality reduction on 8x8 digits | $10 \times 10$ Kohonen MiniSom neural lattice | U-Matrix, BMU Coordinates, Silhouette | Mapped 1,797 handwritten digits to 100 competitive neurons, preserving digit topologies. |
| **1.5: Ensemble Optimization** | Binary breast cancer diagnosis | Random Forest, AdaBoost, Gradient Boosting | Stratified 5-Fold CV, `GridSearchCV` | AdaBoost tuned with $100$ estimators reached top accuracy (**97.72%**). |

---

### [Experiment 2: Energy Models, MCMC & Probabilistic Inference](./Experiment-2/README.md)
Detailed walkthroughs and notebooks are available in [`Experiment-2/`](./Experiment-2/).

| Sub-Experiment | Focus / Task | Architecture / Formulation | Primary Outputs & Diagnostics | Key Findings |
| :--- | :--- | :--- | :--- | :--- |
| **2.1: Deep Belief Networks** | Unsupervised representation learning on 64-dim digits | 2-Layer Bernoulli RBM ($64 \to 32$) + Logistic Regression | Learned $8 \times 8$ receptive filters, Latent classification | Visualized layer 1 & 2 receptive field weights; benchmarked raw vs latent features. |
| **2.2: MCMC Gibbs Sampling** | Sampling from joint Gaussian $\mathcal{N}(\mathbf{0}, \boldsymbol{\Sigma})$ ($\rho = 0.8$) | Iterative conditional sampling: $P(X \mid Y)$ and $P(Y \mid X)$ | Scatter ellipse, Marginal histograms, Trace plots | Empirical $\hat{\rho} = \mathbf{0.7965}$ closely matched theoretical target $\rho = 0.8000$ with zero drift. |
| **2.3: Hidden Markov Models** | Sequential state inference under uncertainty | `CategoricalHMM` ($N=2$ weather states, $M=3$ observations) | Viterbi path decoding, Log likelihood, Sequence generator | Decoded optimal hidden path with log-likelihood of **$-9.9284$**; simulated state trajectories. |

---

## 🛠️ Quickstart & Environment Setup

### 1. Clone the Repository
```bash
git clone https://github.com/Swayam26-rwt/Advanced-Machine-Learning.git
cd Advanced-Machine-Learning
```

### 2. Set Up a Virtual Environment
```bash
# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
# On macOS / Linux:
source .venv/bin/activate
# On Windows:
# .venv\Scripts\activate
```

### 3. Install Required Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Launch Jupyter Notebook
```bash
jupyter notebook
```
Navigate to [`Experiment-1/`](./Experiment-1/) or [`Experiment-2/`](./Experiment-2/) to run and explore any experiment interactively.

---

## 📦 Required Dependencies

All required packages are documented in [`requirements.txt`](./requirements.txt):
- **Scientific Computing**: `numpy`, `pandas`, `scipy`
- **Visualization**: `matplotlib`
- **Machine Learning**: `scikit-learn`
- **Self-Organizing Maps**: `minisom`
- **Hidden Markov Modeling**: `hmmlearn`
- **Interactive Environment**: `jupyter`, `ipykernel`

---

## 📜 License
This repository is open-sourced under the [MIT License](https://opensource.org/licenses/MIT).

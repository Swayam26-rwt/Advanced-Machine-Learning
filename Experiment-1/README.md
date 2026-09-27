# Experiment 1: Advanced Supervised & Unsupervised Learning

This module contains comprehensive experimental implementations covering regularized regression, support vector machines with various kernel transformations, hierarchical clustering paradigms (bottom-up vs. top-down), topological representation learning via Self-Organizing Maps (SOM), and ensemble learning architectures with systematic hyperparameter optimization.

---

## Table of Contents
1. [Module Manifest](#module-manifest)
2. [Experiment 1.1: Regularized Linear Regression](#experiment-11-regularized-linear-regression)
3. [Experiment 1.2: Support Vector Machines (SVM) & Kernel Comparison](#experiment-12-support-vector-machines-svm--kernel-comparison)
4. [Experiment 1.3: Hierarchical Clustering (AGNES vs. DIANA)](#experiment-13-hierarchical-clustering-agnes-vs-diana)
5. [Experiment 1.4: Self-Organizing Maps (Kohonen SOM)](#experiment-14-self-organizing-maps-kohonen-som)
6. [Experiment 1.5: Ensemble Methods & Hyperparameter Optimization](#experiment-15-ensemble-methods--hyperparameter-optimization)
7. [Dependencies & Execution](#dependencies--execution)

---

## Module Manifest

| Notebook | Core Topic | Primary Algorithm(s) | Dataset | Key Metrics / Focus |
| :--- | :--- | :--- | :--- | :--- |
| [`AML_exp1_1.ipynb`](./AML_exp1_1.ipynb) | Regularized Regression | OLS, Ridge ($L_2$), Lasso ($L_1$) | Car Price Dataset | MAE, MSE, RMSE, $R^2$, Coefficient Shrinkage |
| [`AML_exp1_2.ipynb`](./AML_exp1_2.ipynb) | Kernel Methods | Support Vector Classifier (SVC) | Fisher's Iris Dataset | Linear, Poly, RBF, Sigmoid Kernels, Confusion Matrix |
| [`AML_exp1_3.ipynb`](./AML_exp1_3.ipynb) | Hierarchical Clustering | AGNES (Ward) & DIANA (from scratch) | Iris Dataset | Dendrogram, Silhouette Score, Splinter Partitioning |
| [`AML_exp1_4.ipynb`](./AML_exp1_4.ipynb) | Competitive Learning | Kohonen Self-Organizing Map (MiniSom) | Optical Recognition of Handwritten Digits (8x8) | U-Matrix, BMU Coordinates, Silhouette Score |
| [`AML_exp1_5.ipynb`](./AML_exp1_5.ipynb) | Ensemble Methods | Random Forest, AdaBoost, Gradient Boosting | Breast Cancer Wisconsin Diagnostic | Stratified 5-Fold CV, `GridSearchCV` Tuning |

---

## Experiment 1.1: Regularized Linear Regression

### Theoretical Background
Ordinary Least Squares (OLS) minimizes the residual sum of squares:
$$\mathcal{L}_{\text{OLS}}(\mathbf{w}) = \frac{1}{2N} \sum_{i=1}^N \left( y_i - \mathbf{w}^T \mathbf{x}_i \right)^2$$

When features exhibit multicollinearity or when the feature dimension is large, OLS estimates suffer from high variance. Regularization introduces an objective penalty:
- **Ridge Regression ($L_2$ Penalty)**: Adds squared Euclidean norm of coefficients, shrinking parameters toward zero without setting them strictly to zero:
  $$\mathcal{L}_{\text{Ridge}}(\mathbf{w}) = \mathcal{L}_{\text{OLS}}(\mathbf{w}) + \alpha \|\mathbf{w}\|_2^2 = \mathcal{L}_{\text{OLS}}(\mathbf{w}) + \alpha \sum_{j=1}^p w_j^2$$
- **Lasso Regression ($L_1$ Penalty)**: Adds sum of absolute coefficient values, driving coefficients of non-informative features to absolute zero (feature selection):
  $$\mathcal{L}_{\text{Lasso}}(\mathbf{w}) = \mathcal{L}_{\text{OLS}}(\mathbf{w}) + \alpha \|\mathbf{w}\|_1 = \mathcal{L}_{\text{OLS}}(\mathbf{w}) + \alpha \sum_{j=1}^p |w_j|$$

### Methodology & Workflow
1. **Data Preprocessing**: Extracted 14 continuous automotive attributes (`symboling`, `wheelbase`, `carlength`, `carwidth`, `carheight`, `curbweight`, `enginesize`, `boreratio`, `stroke`, `compressionratio`, `horsepower`, `peakrpm`, `citympg`, `highwaympg`).
2. **Train/Test Split**: 80% train, 20% test with standard feature scaling via `StandardScaler`.
3. **Model Training**: Linear Regression, Ridge ($\alpha=1.0$), and Lasso ($\alpha=0.1$, `max_iter=10000`).

### Empirical Results
| Model | MAE ($) | MSE | RMSE ($) | $R^2$ Score |
| :--- | :---: | :---: | :---: | :---: |
| **Linear Regression** | 2,986.84 | $1.6587 \times 10^7$ | 4,072.67 | 0.7798 |
| **Ridge Regression** | **2,963.77** | $\mathbf{1.6352 \times 10^7}$ | **4,043.70** | **0.7830** |
| **Lasso Regression** | 2,985.80 | $1.6579 \times 10^7$ | 4,071.74 | 0.7799 |

- **Observations**: Ridge regression produced the lowest generalization error and highest $R^2$ ($0.7830$), dampening collinear feature variance (such as `enginesize` and `curbweight`).

---

## Experiment 1.2: Support Vector Machines (SVM) & Kernel Comparison

### Theoretical Background
Support Vector Machines identify the optimal separating hyperplane that maximizes the geometric margin $\frac{2}{\|\mathbf{w}\|}$ between classes:
$$\min_{\mathbf{w}, b, \boldsymbol{\xi}} \frac{1}{2}\|\mathbf{w}\|^2 + C \sum_{i=1}^N \xi_i \quad \text{s.t.} \quad y_i(\mathbf{w}^T \phi(\mathbf{x}_i) + b) \ge 1 - \xi_i, \quad \xi_i \ge 0$$

Using the kernel trick, non-linear mappings $\phi(\mathbf{x})$ are evaluated implicitly via kernel functions $K(\mathbf{x}_i, \mathbf{x}_j) = \langle \phi(\mathbf{x}_i), \phi(\mathbf{x}_j) \rangle$:
- **Linear**: $K(\mathbf{x}, \mathbf{z}) = \mathbf{x}^T \mathbf{z}$
- **Polynomial**: $K(\mathbf{x}, \mathbf{z}) = (\gamma \mathbf{x}^T \mathbf{z} + r)^d$
- **Radial Basis Function (RBF)**: $K(\mathbf{x}, \mathbf{z}) = \exp(-\gamma \|\mathbf{x} - \mathbf{z}\|^2)$
- **Sigmoid**: $K(\mathbf{x}, \mathbf{z}) = \tanh(\gamma \mathbf{x}^T \mathbf{z} + r)$

### Methodology & Workflow
- Dataset: Iris Flower dataset (150 samples, 4 morphological features, 3 balanced classes: *Setosa*, *Versicolour*, *Virginica*).
- 80/20 train-test split normalized using `StandardScaler`.
- Trained multi-class SVC across all four kernel variants.

### Performance Summary
| Kernel Variant | Accuracy | Precision (Macro) | Recall (Macro) | F1-Score (Macro) |
| :--- | :---: | :---: | :---: | :---: |
| **Linear** | **1.0000 (100.0%)** | **1.00** | **1.00** | **1.00** |
| **RBF** | **0.9667 (96.7%)** | **0.97** | **0.97** | **0.97** |
| **Polynomial** | 0.9000 (90.0%) | 0.92 | 0.90 | 0.90 |
| **Sigmoid** | 0.9000 (90.0%) | 0.92 | 0.90 | 0.90 |

- **Observations**: Linearly separable boundaries in standardized Iris feature space allow the Linear kernel to reach perfect test classification ($100\%$), with RBF closely behind ($96.7\%$).

---

## Experiment 1.3: Hierarchical Clustering (AGNES vs. DIANA)

### Theoretical Background
Hierarchical clustering organizes data into a nested tree structure (dendrogram):
1. **AGNES (Agglomerative Nesting)**: Bottom-up paradigm. Begins with $N$ singleton clusters and iteratively merges the pair of clusters with minimum distance. Using Ward's linkage:
   $$\Delta \text{ESS}(A, B) = \frac{|A| |B|}{|A| + |B|} \|\mathbf{m}_A - \mathbf{m}_B\|^2$$
2. **DIANA (Divisive Analysis)**: Top-down paradigm. Begins with all samples in a single cluster and iteratively splits the cluster with the largest diameter. At each step, a splinter group is initiated with the point having highest average dissimilarity to the remaining points, iteratively absorbing boundary points.

### Implementation Highlights
- **AGNES**: Scipy hierarchical linkage tree computation and visualization via `dendrogram(linked)`.
- **DIANA (Custom Algorithm)**: Built from first principles in NumPy:
  - Selects cluster $C_k$ with maximal diameter: $D(C) = \max_{\mathbf{x}_i, \mathbf{x}_j \in C} \|\mathbf{x}_i - \mathbf{x}_j\|$.
  - Identifies seed item $\mathbf{x}^* = \arg\max_{\mathbf{x} \in C} \frac{1}{|C|-1} \sum_{\mathbf{y} \in C, \mathbf{y} \ne \mathbf{x}} \|\mathbf{x} - \mathbf{y}\|$.
  - Iteratively reassigns points to splinter group if distance to splinter group is strictly less than distance to remaining group.

### Comparative Results
| Method | Number of Clusters ($k$) | Silhouette Score | Linkage / Splitting Criterion |
| :--- | :---: | :---: | :--- |
| **AGNES** | 3 | 0.4467 | Ward's Minimum Variance Linkage |
| **DIANA** | 3 | **0.4630** | Maximum Diameter Splinter Partitioning |

- **Observations**: DIANA yielded slightly crisper cluster separation (Silhouette score $0.4630$ vs $0.4467$), showing the effectiveness of divisive diameter minimization on the standardized Iris data manifold.

---

## Experiment 1.4: Self-Organizing Maps (Kohonen SOM)

### Theoretical Background
Self-Organizing Maps are unsupervised, competitive artificial neural networks that preserve the topological properties of the input space onto a low-dimensional discrete lattice:
1. **Best Matching Unit (BMU) Competition**:
   $$c = \arg\min_j \|\mathbf{x} - \mathbf{w}_j\|_2$$
2. **Cooperative Weight Update**:
   $$\mathbf{w}_j(t + 1) = \mathbf{w}_j(t) + \eta(t) \cdot h_{cj}(t) \cdot [\mathbf{x} - \mathbf{w}_j(t)]$$
   where $h_{cj}(t) = \exp\left(-\frac{\|\mathbf{r}_c - \mathbf{r}_j\|^2}{2\sigma^2(t)}\right)$ is the neighborhood kernel.

### Experimental Configuration
- **Dataset**: Optical Recognition of Handwritten Digits ($1,797$ instances, $64$ dimensions, $10$ digit classes).
- **Lattice Dimensions**: $10 \times 10$ neuron grid ($100$ topological units).
- **Parameters**: $\sigma = 1.0$, learning rate $\eta = 0.5$, $1,000$ iterations with random weight initialization via `MiniSom`.

### Analysis & Visualizations
- **U-Matrix (Distance Map)**: Highlights boundaries between clusters; high Euclidean distances indicate sparse cluster separation, while low values indicate tight topological clusters.
- **Label Mapping**: Projected ground-truth digit markers onto the $10 \times 10$ lattice, demonstrating clear topological grouping for digits (e.g., distinguishing 0s, 4s, and 6s into distinct regions).
- **Quantization & Clustering Score**: Silhouette score across the 100 BMU cluster assignments is evaluated.

---

## Experiment 1.5: Ensemble Methods & Hyperparameter Optimization

### Theoretical Background
- **Random Forest**: Bagging ensemble of de-correlated decision trees trained on bootstrap samples with random feature sub-spacing. Reduces variance without increasing bias.
- **AdaBoost (Adaptive Boosting)**: Sequential ensemble that iteratively increases sample weights of misclassified instances, minimizing exponential loss.
- **Gradient Boosting**: Sequential ensemble constructing weak learners along the negative gradient of the loss function (pseudo-residuals).

### Methodology
1. **Dataset**: Breast Cancer Wisconsin Diagnostic ($569$ samples, $30$ real-valued features, binary diagnosis).
2. **Validation Strategy**: Stratified 5-Fold Cross Validation.
3. **Hyperparameter Tuning**: Integrated with `scikit-learn` `Pipeline` (`StandardScaler` + Model) into `GridSearchCV`:
   - *Random Forest*: `n_estimators` $\in [50, 100, 200]$, `max_depth` $\in [\text{None}, 5, 10]$, `min_samples_split` $\in [2, 5]$.
   - *AdaBoost*: `n_estimators` $\in [50, 100, 200]$, `learning_rate` $\in [0.01, 0.1, 0.5, 1.0]$.
   - *Gradient Boosting*: `n_estimators` $\in [50, 100, 200]$, `learning_rate` $\in [0.01, 0.1, 0.2]$, `max_depth` $\in [2, 3, 4]$.

### Experimental Benchmark
| Ensemble Model | Baseline 5-Fold CV Accuracy | Post-Tuning Best Accuracy | Optimal Hyperparameters |
| :--- | :---: | :---: | :--- |
| **AdaBoost** | 96.84% $\pm$ 1.31% | **97.72%** | `learning_rate`: 1.0, `n_estimators`: 100 |
| **Gradient Boosting** | 96.31% $\pm$ 2.10% | 96.66% | `learning_rate`: 0.2, `max_depth`: 2, `n_estimators`: 200 |
| **Random Forest** | 95.61% $\pm$ 2.28% | 96.31% | `max_depth`: None, `min_samples_split`: 5, `n_estimators`: 50 |

- **Key Takeaway**: AdaBoost with $100$ estimators and learning rate $1.0$ achieved the highest diagnostic accuracy of **97.72%**, outperforming both Bagging and Gradient Tree Boosting.

---

## Dependencies & Execution
To execute these notebooks:
```bash
pip install -r ../requirements.txt
jupyter notebook
```

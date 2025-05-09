# AIML Assignment 2 – IISc 2025

This repository contains my submission for Assignment 2 of the Artificial Intelligence and Machine Learning course at IISc (March 2025). The assignment consists of three questions involving implementation and analysis of machine learning algorithms on real-world datasets.

---

## Contents

- `AIML_2025_A2_23634.py` – Python code implementing solutions for all three questions.
- `AIML_2025_A2_23634.pdf` – Report detailing the methodology, implementation steps, results, and plots for all attempted questions.
- `AIML_2025_Assignment_2.pdf` – Official assignment sheet containing the problem statements and submission instructions.

---

## Attempted Questions

### Question 1 – Support Vector Machine and Perceptron

- Investigated the separability of the CIFAR-100 dataset using the Perceptron algorithm.
- Implemented soft-margin linear SVMs (primal and dual) using `cvxopt`, and compared runtimes.
- Identified non-separable images using SVM, removed them, and retrained the Perceptron to demonstrate convergence.
- Implemented a kernelized SVM with a Gaussian kernel (C=10, γ=100), achieving 0% misclassification.

### Question 2 – Logistic Regression, MLP, CNN & PCA

- Applied classification techniques to a subset of the MNIST-JPG dataset.
- Trained an MLP on flattened 28x28 images with three hidden layers (512, 256, 128).
- Designed a CNN with two convolutional layers followed by a dense classifier.
- Performed PCA for dimensionality reduction (117 components).
- Retrained MLP and Logistic Regression using PCA features.
- Evaluated all models using confusion matrices, classification metrics (precision, recall, F1-score), and ROC-AUC curves.

### Question 3 – Regression

#### 3.1 Linear and Ridge Regression
- Implemented Ordinary Least Squares (OLS) and Ridge Regression.
- Analyzed performance using MSE on two datasets.
- Discussed rank deficiency and regularization.
- Saved optimal weight vectors to CSV files.

#### 3.2 Support Vector Regression (SVR)
- Used historical stock price data for ticker `BAC`.
- Implemented both linear and kernelized SVR (RBF kernel, γ ∈ {1, 0.1, 0.01, 0.001}) for t ∈ {7, 30, 90}.
- Visualized predicted vs actual stock prices and moving averages across all models.

---
## ⚙️ Requirements

- Python 3.12.8
- Libraries:
  - `numpy`, `pandas`, `matplotlib`, `sklearn`, `torch`, `cvxopt`
---
## 📊 Results Summary

- **Perceptron:** Did not converge on full CIFAR-100 data; converged after removing 403 non-separable images.
- **SVM:** Dual version was slightly faster; kernelized version achieved 100% classification accuracy.
- **MLP vs CNN vs Logistic Regression (with PCA):**
  - CNN achieved highest accuracy (~98%)
  - MLP performed slightly worse but still strong (~95%)
  - Logistic Regression with PCA had acceptable performance (~89%)
- **Regression:**
  - Ridge Regression handled rank-deficient cases better.
  - SVR models showed improved prediction with appropriate kernel and window size tuning.

---
## Contact

For any queries, please reach out to:  
**Kolipaka Bhargav Sashank**  
**SR Number**: 23634  
**Mail :- bhargavsk@iisc.ac.in**  
**IISc, Bangalore**

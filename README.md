# K-Nearest Neighbors (KNN) - Interview Revision Notes

## 1. What is KNN?
- KNN is a **supervised, non-parametric, lazy learning** algorithm.
- To predict for a new point:
  1. Compute its **distance** to every training point (Euclidean by default).
  2. Pick the **k closest** points (its neighbours).
  3. **Classification:** majority vote of the k labels. **Regression:** average of the k values.
- Example: k = 3, neighbours are (A, A, B) -> predict **A**.
- **Lazy** = no training phase. It just stores the data; all work happens at prediction time.
- Common distances: Euclidean, Manhattan, Minkowski, Cosine.

## 2. How to choose k (k is a hyperparameter)

### a) Heuristic
- Rule of thumb: **k ~ sqrt(n)**, where n = number of training samples.
- Prefer an **odd k** for binary classification to avoid ties.
- Only a starting point, not the final answer.

### b) Experimentation: Cross-Validation (preferred)
1. Keep a **test set aside**; never use it for choosing k.
2. Split the remaining data into **K folds** (e.g. 5).
3. For each candidate k (1, 3, 5, 7, ...):
   - Train on K-1 folds, validate on the remaining fold.
   - Repeat for every fold and **average the accuracy**.
4. Pick the k with the **highest average CV accuracy**.
5. Finally, evaluate once on the test set.

> Correction: you don't just do one train/test split. CV repeats the split across folds, and the test set stays untouched until the end.

## 3. How to draw the decision surface (2 features)
1. Plot the training data as a scatter plot (colour by class).
2. Find min and max of each feature (add a small margin) to get the x and y range.
3. Create a grid with `np.meshgrid` over that range.
4. Flatten the grid into points (`np.c_[xx.ravel(), yy.ravel()]`).
5. Run `model.predict` on every grid point.
6. Reshape the predictions to the grid shape.
7. Colour the regions with `plt.contourf`, then plot the training points on top.

```python
import numpy as np
import matplotlib.pyplot as plt

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

Z = knn.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3)
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor="k")
plt.show()
```

## 4. Overfitting and Underfitting in KNN

| | Small k (e.g. k = 1) | Large k (e.g. k near n) |
|---|---|---|
| Fit | **Overfitting** | **Underfitting** |
| Boundary | Jagged, complex | Very smooth |
| Bias / Variance | Low bias, **high variance** | High bias, **low variance** |
| Behaviour | Memorises noise and outliers | Ignores local patterns |
| Extreme | k = 1: training accuracy is 100% | k = n: always predicts the majority class |

- Best k sits in the middle, found by cross-validation.
- Higher k = smoother boundary, more generalisation, but too high loses real patterns.

## 5. Limitations of KNN

| Limitation | Why | Fix |
|---|---|---|
| **Large datasets** | Lazy learner: every prediction compares against all n points (O(n·d) per query), and all data must be stored. Slow at prediction time. | KD-tree, Ball tree, Approximate NN (FAISS), sampling |
| **High-dimensional data** | Curse of dimensionality: distances become almost equal, so "nearest" loses meaning. | PCA, feature selection |
| **Outliers / noise** | With small k, a few noisy neighbours can flip the prediction. | Larger k, remove outliers |
| **Non-homogeneous scales** | Large-scale features dominate the distance (e.g. salary vs age). | **Feature scaling** (StandardScaler / MinMax) |
| **Imbalanced data** | The majority class dominates the vote. | Distance-weighted KNN, resampling (SMOTE), class weights |

## 6. Quick Facts
- KNN has **no learned parameters**; k and the distance metric are hyperparameters.
- **Always scale features** before KNN.
- Works for both classification and regression.
- Weighted KNN: closer neighbours get more vote weight (`weights="distance"` in sklearn).
- Training time: ~0. Prediction time: high.

## 7. Common Interview Questions
- **Why is KNN called lazy?** It doesn't learn a model during training; it only stores the data.
- **Why scale features?** Distance is scale-sensitive; unscaled features distort neighbours.
- **What happens at k = 1? At k = n?** Overfit, jagged boundary / underfit, always majority class.
- **How do you pick k?** sqrt(n) as a start, then cross-validation.
- **Why does KNN fail in high dimensions?** Distances concentrate, so all points look equally far.
- **How do you speed up KNN?** KD-tree, Ball tree, approximate nearest neighbours, dimensionality reduction.

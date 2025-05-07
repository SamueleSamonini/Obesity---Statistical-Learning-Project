# Obesity - Statistical Learning Project

This project applies statistical learning techniques to a dataset on obesity, aiming to uncover patterns and build predictive models using both unsupervised and supervised approaches.

## 📊 Dataset

The dataset includes 2,111 individuals from Colombia, Peru, and Mexico. It contains biometric and lifestyle variables such as age, weight, height, physical activity, food habits, and water intake. The target variable is the obesity level, categorized into 7 classes.

## 🔍 Project Structure

### 1. **Unsupervised Learning**

- **EDA**: Visual exploration of numeric and categorical variables.
- **PCA**: Dimensionality reduction to understand variable importance and structure.
- **K-means Clustering**: Optimal number of clusters (k = 7) selected using the silhouette method.
- **Cluster Profiling**: Analysis of behavior and age distribution across clusters.

### 2. **Supervised Learning**

- **Models Tested**:
  - Support Vector Machine (SVM)
  - Classification and Regression Tree (CART)
  - Random Forest (RF)
  - Multinomial Logistic Regression
  - Gradient Boosting (GBM)
  - Neural Network (nnet)

- **Evaluation Metrics**:
  - Accuracy
  - Kappa statistic
  - F1 Macro
  - RMSE
  - ROC and AUC curves

## 🧠 Methodology

- Data preprocessing (dummy variables, normalization, outlier check).
- Feature importance analysis for each model.
- Training and testing split (60/40) with cross-validation.
- Visualization of confusion matrices, ROC curves, and model comparison.

## ✅ Results

- **Best performing model**: Random Forest
  - Accuracy: 95.01%
  - Kappa: 0.94
  - F1 Macro: 0.99
  - RMSE: 0.25
- Logistic Regression and GBM also performed strongly.
- CART and SVM showed weaker results, especially in middle obesity levels.

## 📁 Files

- `unsupervised_analysis.R`: EDA, PCA, K-means clustering, and visualizations.
- `supervised_analysis.R`: Model training, evaluation, and comparison.
- `data/obesity.csv`: Input dataset (if included).
- `Report.pdf`: Full written report of the project and findings.

## 📌 Conclusion

This project highlights how combining unsupervised and supervised methods provides deep insights into complex health-related datasets, supporting early detection and personalized recommendations.

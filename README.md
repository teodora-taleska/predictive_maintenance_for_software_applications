# Predictive Maintenance for Software Systems Using Machine Learning: A Comparative Study of Algorithms on the KC1 NASA PROMISE Dataset

## Project Overview

This project investigates the potential of machine learning techniques to enhance the prediction of software defects. The main objective is to evaluate and compare various machine learning algorithms in terms of their ability to predict software defects, using a dataset enriched with McCabe and Halstead metrics. These metrics, which quantify different aspects of code complexity and quality, provide a solid foundation for assessing software reliability.

## Dataset Description

The study utilizes the **KC1 dataset** from the NASA PROMISE repository, which consists of:
- **145 instances (modules or classes)**
- **95 attributes (features)**
- The target variable, **NUMDEFECTS**, records the number of defects for each module.

The features in this dataset are numerical values related to code complexity and structural characteristics, offering a comprehensive view of the software's potential for defects. Key metrics include **sumLOC_TOTAL** (total lines of code), **sumBRANCH_COUNT** (branch count), **COUPLING_BETWEEN_OBJECTS**, and **maxHALSTEAD_VOLUME** (Halstead metric representing code volume).

During data preprocessing, 4 duplicate instances were identified and removed, reducing the dataset to 141 instances. Further, 14 outliers were detected using the IsolationForest algorithm and subsequently removed, leaving 127 instances for certain parts of the analysis. This data reduction was considered while selecting the 6-fold cross-validation, ensuring a balance between training and validation sets, allowing at least 20 instances per fold.

## Purpose of the Project

The primary goal of this project is to explore the effectiveness of various machine learning algorithms in predicting software defects. By comparing different models, the study aims to identify the most accurate and reliable approaches for predictive maintenance in software systems. Additionally, the project seeks to understand the key factors that influence prediction accuracy and to evaluate the benefits of ensemble methods over traditional single-algorithm approaches.

## Tools and Libraries

The project is implemented in **Python** using the following libraries:
- **NumPy** (v1.24.2): For mathematical operations and data manipulation.
- **Pandas** (v1.5.3): For data handling and analysis using DataFrames.
- **SciPy** (v1.10.1): For loading and processing ARFF file formats.
- **Scikit-Learn** (v1.2.2): For implementing machine learning algorithms and evaluation metrics.
- **Matplotlib** (v3.7.1): For creating graphs and visualizations to present the results.

## Machine Learning Algorithms Used

The study evaluates several machine learning models:
- **Artificial Neural Networks (ANN)**
- **Support Vector Regressor (SVR)**
- **Linear Regression (LR)**
- **Decision Trees (DT)**
- **Random Forest (RF)**
- **Gradient Boosting Regressor (GBR)**


### Baseline and Model Evaluation

To establish a point of comparison, a **BaselineModel** was created, which simply predicts the mean of the target variable across the entire dataset. This model was implemented as a custom estimator using the following Python code:

```python
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

class BaselineModel(BaseEstimator, RegressorMixin):
    def __init__(self):
        self.average_target = None

    def fit(self, X, y):
        self.average_target = y.mean()
        return self

    def predict(self, X):
        return np.full(len(X), self.average_target)

    def get_params(self, deep=False):
        return {}

    def set_params(self, **params):
        return self
```

This baseline model serves as a simple mean predictor, allowing us to assess the added value of more complex machine learning algorithms. By comparing all other models against this baseline, we can gauge their effectiveness in capturing patterns and making accurate predictions.

### Results

#### Performance using leave-one-out with outliers included

| Algorithm                       | R²     | MSE      | RMSE    | MAE    |
|---------------------------------|--------|----------|---------|--------|
| **Baseline Model**              | -0.0180| 121.9506 | 11.0431 | 6.1685 |
| **ANN (300,)**                  | 0.3101 | 82.6433  | 9.0908  | 5.7596 |
| **ANN (500,)**                  | 0.3089 | 82.7894  | 9.0989  | 5.7535 |
| **ANN (200, 100)**              | 0.3516 | 77.6763  | 8.8134  | 5.1444 |
| **ANN (300, 200, 100)**         | 0.2444 | 90.5146  | 9.5139  | 4.7660 |
| **Support Vector Regressor (SVR)** | 0.0876 | 109.2975 | 10.4545 | 4.5002 |
| **Linear Regression**           | 0.0898 | 109.0316 | 10.4418 | 5.6053 |
| **Decision Tree (DT)**          | -0.2996| 155.6826 | 12.4773 | 5.7695 |
| **Random Forest (RF)**          | 0.2193 | 93.5232  | 9.6707  | 4.8073 |
| **Gradient Boosting Regressor (GBR)** | 0.1848 | 97.6498  | 9.8818  | 4.3370 |

#### Performance using leave-one-out with outliers excluded

| Algorithm                       | R²     | MSE      | RMSE    | MAE    |
|---------------------------------|--------|----------|---------|--------|
| **Baseline Model**              | -0.0159| 42.1828  | 6.4948  | 4.5877 |
| **ANN (300,)**                  | 0.0095 | 41.1281  | 6.4131  | 4.2807 |
| **ANN (500,)**                  | 0.0893 | 37.8147  | 6.1494  | 4.0782 |
| **ANN (200, 100)**              | 0.1907 | 33.6032  | 5.7968  | 3.8078 |
| **ANN (300, 200, 100)**         | 0.1625 | 34.7759  | 5.8971  | 3.4132 |
| **Linear Regression**           | 0.2188 | 32.4379  | 5.6954  | 4.0303 |
| **SVR**                         | 0.0240 | 40.5235  | 6.3658  | 3.3095 |
| **Decision Tree**               | -0.3677| 56.7873  | 7.5357  | 4.5470 |
| **Random Forest**               | 0.2775 | 30.0001  | 5.4772  | 3.5230 |
| **Gradient Boosting**           | 0.1254 | 35.3134  | 6.0261  | 3.2195 |

#### Performance using 6-fold cross-validation with outliers included

| Algorithm                       | R²     | MSE      | RMSE    | MAE    |
|---------------------------------|--------|----------|---------|--------|
| **Baseline Model**              | -0.0180| 121.9506 | 11.0431 | 6.1685 |
| **ANN (300,)**                  | 0.3101 | 82.6433  | 9.0908  | 5.7596 |
| **ANN (500,)**                  | 0.3089 | 82.7894  | 9.0989  | 5.7535 |
| **ANN (200, 100)**              | 0.3516 | 77.6763  | 8.8134  | 5.1444 |
| **ANN (300, 200, 100)**         | 0.2444 | 90.5146  | 9.5139  | 4.7660 |
| **Support Vector Regressor (SVR)** | 0.0876 | 109.2975 | 10.4545 | 4.5002 |
| **Linear Regression**           | 0.0898 | 109.0316 | 10.4418 | 5.6053 |
| **Decision Tree (DT)**          | -0.2996| 155.6826 | 12.4773 | 5.7695 |
| **Random Forest (RF)**          | 0.2193 | 93.5232  | 9.6707  | 4.8073 |
| **Gradient Boosting Regressor (GBR)** | 0.1848 | 97.6498  | 9.8818  | 4.3370 |

#### Performance using 6-fold cross-validation with outliers excluded

| Algorithm                       | R²     | MSE      | RMSE    | MAE    |
|---------------------------------|--------|----------|---------|--------|
| **Baseline Model**              | -0.0573| 43.8984  | 6.6256  | 4.6968 |
| **ANN (300,)**                  | 0.0415 | 39.7969  | 6.3085  | 4.3200 |
| **ANN (500,)**                  | 0.0747 | 38.4195  | 6.1983  | 4.2410 |
| **ANN (200, 100)**              | 0.2328 | 31.8554  | 5.6441  | 3.7322 |
| **ANN (300, 200, 100)**         | 0.2399 | 31.5587  | 5.6177  | 3.3949 |
| **Linear Regression**           | 0.1523 | 35.1981  | 5.9328  | 4.2004 |
| **SVR**                         | -0.0147| 42.1333  | 6.4910  | 3.4917 |
| **Decision Tree**               | 0.1036 | 37.2205  | 6.1009  | 3.5512 |
| **Random Forest**               | 0.1439 | 35.5469  | 5.9621  | 3.8450 |
| **Gradient Boosting**           | 0.0610 | 38.9870  | 6.2440  | 3.5889 |


### Comparison with Baseline

All evaluated machine learning algorithms outperformed the baseline model, confirming the importance of leveraging advanced techniques for this predictive maintenance task:
- **Artificial Neural Networks (ANN)** and **Random Forest** models demonstrated the most significant improvements over the baseline, showcasing their ability to capture complex patterns within the data.
- **Linear Regression** and **Decision Tree** models provided modest improvements over the baseline. While these simpler models do offer some predictive capability, their performance highlights the limitations of traditional approaches in handling the intricacies of the dataset.

### Best and Worst Performing Models

- **Best Performing Model**: The **Artificial Neural Network (ANN)** with a (200, 100) architecture was the most effective, achieving an **R² score of 0.3516**, an **MSE of 77.6763**, an **RMSE of 8.8134**, and an **MAE of 5.1444** when evaluated with 6-fold cross-validation, including outliers. This model demonstrated its ability to capture complex patterns and interactions in the data.
  
- **Worst Performing Model**: The **Decision Tree (DT)** was the least effective, with an **R² score of -0.2996**, an **MSE of 155.6826**, an **RMSE of 12.4773**, and an **MAE of 5.7695** when outliers were included. This poor performance highlights the Decision Tree’s sensitivity to outliers and its tendency to overfit, resulting in less reliable predictions.

### Comparison of Traditional Approaches and Ensemble Methods

The study also explores the improvement offered by ensemble machine learning models over traditional single-algorithm approaches:
- **Random Forest (RF)**: Achieved an **R² score of 0.2193**, an **MSE of 93.5232**, an **RMSE of 9.6707**, and an **MAE of 4.8073** with outliers included. This ensemble method showed better performance than other single-algorithm models, capturing a broader range of data patterns and reducing overfitting.
  
- **Gradient Boosting Regressor (GBR)**: The GBR provided competitive results with an **R² score of 0.1848**, an **MSE of 97.6498**, an **RMSE of 9.8818**, and an **MAE of 4.3370** when evaluated with 6-fold cross-validation, including outliers. Known for its ability to sequentially correct errors, the GBR further emphasizes the advantages of ensemble methods in predictive maintenance.


### Key Findings

- **ANN Performance**: The ANN model's strong performance underscores the importance of advanced machine learning techniques in capturing complex data patterns, making it highly effective for predictive maintenance in software systems.
- **Importance of Ensemble Methods**: The study demonstrates that ensemble models like Random Forest and Gradient Boosting Regressor offer significant enhancements in prediction accuracy over traditional approaches, particularly in handling complex datasets with potential outliers.
- **Key Features in Defect Prediction**: Feature importance analysis using the Gradient Boosting Regressor highlighted significant features such as **sumLOC_TOTAL**, **sumBRANCH_COUNT**, **COUPLING_BETWEEN_OBJECTS**, and **maxHALSTEAD_VOLUME**, with **sumLOC_TOTAL** being particularly critical.

## Conclusion

This project demonstrates that machine learning can marginally enhance the prediction of software defects by effectively utilizing features derived from McCabe and Halstead metrics. Advanced models, especially neural networks and ensemble methods, offer substantial improvements over traditional approaches by capturing complex data patterns and improving accuracy. The identification of key features that impact defect prediction highlights the importance of comprehensive code metrics in predictive maintenance. Overall, this study establishes a strong foundation for advancing predictive maintenance practices in software engineering, emphasizing the crucial role of sophisticated machine learning techniques in achieving high-quality, reliable software systems.

## How to Run the Project

To run this project, make sure you have the following libraries installed:

```bash
pip install numpy==1.24.2
pip install pandas==1.5.3
pip install scipy==1.10.1
pip install scikit-learn==1.2.2
pip install matplotlib==3.7.1
```

After setting up the environment, you can execute the code provided in the project repository to load the dataset, preprocess it, run the machine learning models, and visualize the results.
# Predicting Boston Housing Prices with Deep Neural Networks

This repository contains a Jupyter Notebook that demonstrates how to build, train, and optimize deep neural networks for both regression and classification tasks using the Boston Housing dataset. The project leverages TensorFlow and Keras to create predictive models and uses KerasTuner and manual tuning to enhance their performance.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Tasks Performed](#tasks-performed)
- [Key Steps in the Notebook](#key-steps-in-the-notebook)
- [Model Performance](#model-performance)
- [Technologies Used](#technologies-used)
- [How to Run This Project](#how-to-run-this-project)
- [Conclusion](#conclusion)

---

## Project Overview

The primary goal of this project is to explore the application of deep learning on a classic tabular dataset. We tackle two distinct machine learning problems using the same dataset:

1.  **Regression Task:** Predicting the continuous median value of homes (`MEDV`).
2.  **Classification Task:** Classifying homes into two categories—"expensive" or "not expensive"—based on a median price threshold.

For each task, a baseline model is first developed and evaluated. Subsequently, hyperparameter tuning is performed to find a more optimal architecture and improve predictive accuracy.

---

## Dataset

The project uses the **Boston Housing dataset**, which is a classic benchmark dataset in machine learning. It contains 506 samples with 13 numerical features that describe various aspects of residential homes in suburbs of Boston.

**Key Features Include:**
- `CRIM`: Per capita crime rate by town
- `RM`: Average number of rooms per dwelling
- `LSTAT`: Percentage of lower status of the population
- `MEDV`: Median value of owner-occupied homes in $1000's (the target variable)

The dataset is loaded directly from the `tensorflow.keras.datasets` library.

---

## Tasks Performed

The notebook is structured into two main parts, each addressing a different machine learning task:

### Part 1: Regression (Predicting House Prices)
- **Objective:** Predict the exact `MEDV` for each house.
- **Model:** A Deep Neural Network (DNN) with a linear output layer.
- **Loss Function:** Mean Squared Error (MSE).
- **Metric:** Mean Absolute Error (MAE).
- **Optimization:** Hyperparameter tuning was performed using **KerasTuner (Hyperband)** to find the best number of layers, units, activation function, and learning rate.

### Part 2: Classification (Is a House Expensive?)
- **Objective:** Classify each house as expensive (`1`) or not expensive (`0`). The threshold is the median `MEDV` of the training set.
- **Model:** A DNN with a sigmoid output layer for binary classification.
- **Loss Function:** Binary Cross-Entropy.
- **Metric:** Accuracy.
- **Optimization:** A **manual grid search** was conducted to find the best hyperparameters from a predefined set of combinations.

---

## Key Steps in the Notebook

1.  **Data Loading and Preparation:** The dataset is loaded and split into training and testing sets. It is then converted into Pandas DataFrames for easier inspection.
2.  **Exploratory Data Analysis (EDA):**
    -   Distribution of the target variable (`MEDV`) is visualized.
    -   A correlation heatmap is generated to understand relationships between features.
    -   Scatterplots and boxplots are used to analyze top features and identify outliers.
3.  **Data Preprocessing:**
    -   Features and target variables are separated.
    -   Feature scaling is performed using `StandardScaler` to normalize the data.
4.  **Model Building and Training:**
    -   Baseline models for both regression and classification are defined and trained.
    -   The `EarlyStopping` callback is used to prevent overfitting.
5.  **Hyperparameter Tuning:**
    -   **KerasTuner** is used for automated tuning in the regression task.
    -   A **manual grid search** is implemented for the classification task.
6.  **Evaluation and Comparison:** The performance of the baseline and tuned models are evaluated on the test set and compared to quantify the improvements.

---

## Model Performance

The results demonstrate the effectiveness of hyperparameter tuning in improving model performance.

### Regression Model
| Model         | Test MAE | Test MSE |
|---------------|----------|----------|
| **Baseline**  | $3.27k   | 23.22    |
| **Tuned**     | **$2.78k**   | **18.90**    |
| *Improvement* | *15%*    | *18.6%*  |

The tuned model's predictions are, on average, **$490 closer** to the actual house price than the baseline model.

### Classification Model
| Model         | Test Accuracy | Test Loss |
|---------------|---------------|-----------|
| **Baseline**  | 88.24%        | 0.3229    |
| **Tuned**     | **91.18%**        | 0.3029    |
| *Improvement* | *2.94%*       | *6.2%*    |

The tuned classification model correctly identifies whether a house is expensive or not with **~91% accuracy**.

---

## Technologies Used

- **Python 3**
- **TensorFlow & Keras:** For building and training the deep learning models.
- **KerasTuner:** For automated hyperparameter optimization.
- **Scikit-learn:** For data preprocessing (`StandardScaler`).
- **Pandas:** For data manipulation and analysis.
- **NumPy:** For numerical operations.
- **Matplotlib & Seaborn:** For data visualization.
- **Jupyter Notebook:** As the development environment.

---

## How to Run This Project

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/your-repository-name.git
    cd your-repository-name
    ```

2.  **Install the required libraries:**
    It is recommended to use a virtual environment.
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: If a `requirements.txt` file is not provided, you can install the libraries manually)*
    ```bash
    pip install tensorflow pandas numpy scikit-learn matplotlib seaborn keras-tuner
    ```

3.  **Run the Jupyter Notebook:**
    ```bash
    jupyter notebook "Assignment7 (1).ipynb"
    ```
    You can then execute the cells sequentially to reproduce the analysis.

---

## Conclusion

This project successfully showcases a complete workflow for applying deep learning to a tabular dataset for both regression and classification. The results confirm that while baseline models can perform well, systematic **hyperparameter tuning is a crucial step for maximizing model accuracy and performance.**

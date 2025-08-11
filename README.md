# Composer Identification using Deep Learning

This project presents a deep learning solution for automatically identifying the composer of classical music pieces from MIDI files. Using a dataset of works by Bach, Beethoven, Chopin, and Mozart, we build and optimize two powerful neural network architectures—a **Convolutional Neural Network (CNN)** and a **Long Short-Term Memory (LSTM)** network—to achieve high classification accuracy.

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Project Workflow](#project-workflow)
- [Key Findings from EDA](#key-findings-from-eda)
- [Model Architectures](#model-architectures)
  - [1. 1D Convolutional Neural Network (CNN)](#1-1d-convolutional-neural-network-cnn)
  - [2. Long Short-Term Memory (LSTM) Network](#2-long-short-term-memory-lstm-network)
- [Technologies Used](#technologies-used)
- [How to Run This Project](#how-to-run-this-project)
- [Conclusion](#conclusion)

---

## Problem Statement

Classifying the composer of a classical music piece is a challenging task, even for experienced musicians. Stylistic nuances between composers like Bach, Beethoven, Chopin, and Mozart are often subtle and require deep domain knowledge. This project aims to automate this process by developing a robust deep learning model that can accurately predict the composer from a MIDI file.

**Objective:** To build and evaluate CNN and LSTM models that achieve high accuracy, precision, and recall, providing a reliable tool for automated music analysis and classification.

---

## Dataset

The project uses the "Classical Music MIDI" dataset available on Kaggle. It contains thousands of MIDI files categorized by four renowned composers:
- **Bach**
- **Beethoven**
- **Chopin**
- **Mozart**

**Link to Dataset:** [MIDI Classic Music on Kaggle](https://www.kaggle.com/datasets/blanderbuss/midi-classic-music)

---

## Project Workflow

The project follows a systematic machine learning pipeline:

1.  **Data Loading:** MIDI files for the four target composers are loaded and organized.
2.  **Feature Extraction:** Musical features are extracted from each MIDI file using the `pretty_midi` library. Key features include:
    -   Note Density & Polyphony
    -   Pitch Mean & Variance
    -   Velocity Mean & Variance
    -   Note Duration Mean & Variance
    -   Estimated Tempo
3.  **Exploratory Data Analysis (EDA):** The extracted features are visualized to understand distributions, identify class imbalances, and find correlations between features.
4.  **Data Preprocessing:**
    -   The dataset is split into training and testing sets.
    -   **SMOTE (Synthetic Minority Over-sampling TEchnique)** is applied to the training data to address significant class imbalance.
    -   Features are normalized using `StandardScaler`.
5.  **Model Building & Training:** Both CNN and LSTM architectures are designed, trained, and optimized.
6.  **Evaluation:** Models are evaluated using accuracy, classification reports, and confusion matrices to assess their performance on unseen data.

---

## Key Findings from EDA

-   **Class Imbalance:** The original dataset was heavily skewed, with the 'Bach' class being significantly over-represented. This justified the use of **SMOTE** to create a balanced training set.
-   **Feature Separability:** Box plots and violin plots revealed distinct stylistic patterns. For example:
    -   **Bach:** Higher and more consistent note velocities.
    -   **Beethoven:** Greater polyphony and a higher number of instruments.
    -   **Mozart:** Lower variance in note durations.
-   **Feature Correlation:** A correlation heatmap showed moderate relationships (e.g., between tempo and duration mean), but no severe multicollinearity, indicating that all extracted features provided unique information.

---

## Model Architectures

Two deep learning architectures were implemented to tackle this classification problem.

### 1. 1D Convolutional Neural Network (CNN)

-   **Rationale:** 1D CNNs are excellent at detecting local patterns and motifs within sequential data. In this context, they can identify characteristic combinations of musical features that define a composer's style.
-   **Architecture:** A "pyramid-style" CNN was designed with three convolutional blocks of increasing filter sizes (32 -> 64 -> 128) to learn hierarchical features. `BatchNormalization` and `Dropout` were used for regularization.

### 2. Long Short-Term Memory (LSTM) Network

-   **Rationale:** LSTMs are a type of Recurrent Neural Network (RNN) specifically designed to capture long-range dependencies in sequential data. This makes them suitable for understanding the temporal structure of musical features.
-   **Architecture:** A stacked LSTM model with two layers (64 and 32 units) was implemented, followed by a dense classifier head. `Dropout` was used to prevent overfitting.

---

## Technologies Used

-   **Python 3**
-   **TensorFlow & Keras:** For building, training, and optimizing the deep learning models.
-   **KerasTuner:** For automated hyperparameter tuning.
-   **Scikit-learn:** For data splitting, encoding, and performance metrics.
-   **pretty_midi:** For parsing MIDI files and extracting musical features.
-   **imblearn:** For implementing the SMOTE algorithm to handle class imbalance.
-   **Pandas & NumPy:** For data manipulation and numerical operations.
-   **Matplotlib & Seaborn:** For data visualization.
-   **Google Colab:** As the development environment.

---

## How to Run This Project

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/jas-usd/composer-identification.git
    ```

2.  **Set Up the Environment:**
    This project was developed in Google Colab. To run it locally, ensure you have the required libraries installed:
    ```bash
    pip install tensorflow keras-tuner scikit-learn pretty_midi imblearn pandas matplotlib seaborn
    ```

3.  **Download the Dataset:**
    -   Download the dataset from [Kaggle](https://www.kaggle.com/datasets/blanderbuss/midi-classic-music).
    -   Unzip the file and place the `midiclassics` folder in the root directory of this project. The notebook expects the following directory structure:
        ```
        .
        ├── project_notebook_team1.py
        └── midiclassics/
            ├── Bach/
            ├── Beethoven/
            ├── Chopin/
            └── Mozart/
        ```

4.  **Run the Notebook:**
    -   Open and run the `Project_Notebook_Team1.ipynb` notebook in a Jupyter environment or Google Colab.
    -   Execute the cells sequentially to perform data loading, feature extraction, training, and evaluation.

---

## Conclusion

Both models performed well after optimization, but the **Tuned CNN showed a slight advantage** in overall performance and stability.

| Model              | Final Test Accuracy | Key Observation                                       |
| ------------------ | ------------------- | ----------------------------------------------------- |
| **Tuned CNN**      | **84.1%**           | Higher precision on difficult classes (e.g., Beethoven). |
| **Optimized LSTM** | 83.7%               | Strong recall but slightly lower precision.           |

For this feature set, the 1D-CNN architecture was marginally more effective at capturing the discriminative patterns in the musical data, making it the recommended model. By addressing the critical issue of class imbalance with SMOTE and applying robust regularization techniques, our tuned CNN model achieved an impressive accuracy of **84.1%**. The results highlight the potential of AI to automate complex musicological tasks and make music analysis more accessible.

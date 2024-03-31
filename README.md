# Quantum ML vs Classical ML Research

## Introduction

Quantum computing is anticipated to revolutionize the tech industry with its potential to tackle computational problems that are intractable for classical computers. Many speculate that quantum computers could offer advantages over classical computers in fields such as finance, machine learning, cryptography, chemistry, optimization, among others. This paper presents an experimental research conducted by Vanderbilt University personnels, comparing the performance of quantum machine learning (QML) algorithms to classical machine learning (ML) algorithms. It evaluates metrics such as precision score, recall score, F1 score, and runtime across multiple datasets. The research compared several classical machine learning algorithms—Logistic Regression (LR), Naive Bayes (NB), Support Vector Classifiers (SVC), Decision Tree (DT), and Random Forest (RF)—with quantum machine learning algorithms, such as Variational Quantum Classifiers (VQC), Quantum Support Vector Classifier (QSVC), and Pegasos QSVC, using four different datasets. Then, a comprehensive analysis explored how the runtime of specific algorithms changed with the number of features and observations across the four datasets. This secondary analysis provided a direct comparison of models within the SVC family models (i.e., between classical SVC and QSVC algorithms). It also examined VQC to investigate potential trends pertaining to the relationship between runtime and number of features/observations.

## Data

Here is a list of the different data sets that were used for the research work. 

- Predictive maintenance: https://www.kaggle.com/datasets/shivamb/machine-predictive-maintenance-classification/data
- Raisin: https://www.kaggle.com/datasets/nimapourmoradi/raisin-binary-classification
- Room: https://www.kaggle.com/datasets/sachinsharma1123/room-occupancy
- Diabetes: https://www.kaggle.com/datasets/girishvutukuri/diabetes-binary-classification

## Methdology

- The experiment was conducted using Vanderbilt University's ACCRE cloud computing resource, with datasets and results organized systematically on GitHub.
- Datasets were split into training and testing sets (80/20 ratio), with exploratory data analysis (EDA) used to balance class distributions and identify key features.
- Five classical ML algorithms were trained using grid-search for hyperparameter tuning, while QML algorithms were trained without hyperparameter tuning due to long training times.
- Optimal models for both classical and QML algorithms were selected based on test performance and runtime, with the less complex model preferred in cases of similar performance.
- A comparative analysis of runtimes between classical and quantum models was conducted to gauge how feasible QML algos are for adaptation in the real world.
- After extracting all the experiment’s results in .csv formats, the results were loaded into Tableau for data visualization. 

## Algorithms

Classical Machine Learning Algorithms
- Logistic Regression
- Naive Bayes
- Random Forest Classifier
- Decision Tree Classifier
- Support Vector Classifier
  
Quantum Machine Learning Algorithms
- Quantum Support Vector Classifier
- Pegasos Quantum Support Vector Classifier
- Variational Quantum Classifier
  
## Tools & Technologies 

- Python
- ACCRE
- Tableau
  
## References

-

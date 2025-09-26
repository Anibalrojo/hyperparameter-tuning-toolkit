# Hyperparameter Tuning Toolkit

A comprehensive educational project designed to compare and contrast various hyperparameter tuning techniques for machine learning models. This repository serves as a learning resource for data scientists and machine learning practitioners.

## Educational Purpose

This toolkit is primarily created for **learning purposes** and aims to provide:

* Practical implementations of different hyperparameter optimization methods
* Clear comparisons of their performance, efficiency, and use cases
* Hands-on examples with real datasets to demonstrate concepts in action
* Code that is well-documented and easy to follow for educational purposes

As you explore this repository, you'll gain practical experience with hyperparameter tuning techniques that are essential for optimizing machine learning models in real-world applications.

## Project Goal

The primary goal of this project is to provide a clear, hands-on demonstration of different hyperparameter optimization methods, including:

* **Grid Search** - Exhaustive search through a manually specified parameter space
* **Random Search** - Randomly sampling from the parameter space, often more efficient than grid search
* **Bayesian Optimization (using Optuna)** - Sequential model-based optimization using surrogate models
* **Genetic Algorithms (using DEAP)** - Evolutionary approach to hyperparameter optimization

This toolkit serves as an educational resource to understand the trade-offs in terms of performance, computational cost, and implementation complexity for each technique.

## Repository Structure

```
hyperparameter-tuning-toolkit/
├── README.md           # You are here.
├── requirements.txt    # Project dependencies.
├── .gitignore          # Files to ignore in git.
|
├── data/
│   ├── 01_raw/         # Raw, immutable data.
│   │   ├── breast_cancer.csv
│   │   └── pima-indians-diabetes.csv
│   └── 02_processed/   # Cleaned data for modeling.
│       ├── cancer_X_test.csv
│       ├── cancer_X_train.csv
│       ├── cancer_y_test.csv
│       ├── cancer_y_train.csv
│       ├── pima_X_test.csv
│       ├── pima_X_train.csv
│       ├── pima_y_test.csv
│       └── pima_y_train.csv
|
├── notebooks/          # Jupyter notebooks for exploration and presentation.
│   ├── 01_pima_diabetes_grid_random_bayesian.ipynb
│   └── 02_breast_cancer_genetic_algorithm.ipynb
|
├── src/                # Source code for use in this project.
│   ├── __init__.py     # Makes src a Python module
│   ├── data/           # Scripts to download or generate data
│   │   ├── __init__.py
│   │   └── make_dataset.py
│   ├── features/       # Scripts to transform data for modeling
│   │   └── build_features.py
│   ├── models/         # Scripts to train models
│   │   └── __init__.py
│   └── visualization/  # Scripts to create visualizations
│       ├── __init__.py
│       └── visualize.py
|
└── tests/              # Unit tests
    ├── __init__.py
    └── test_data.py
```

## How to Get Started

### 1. Clone the Repository

```bash
git 
cd hyperparameter-tuning-toolkit
```

### 2. Create a Virtual Environment

It is highly recommended to use a virtual environment to manage dependencies.

```bash
# For Windows
python -m venv venv
.\venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the Data

Run the data preparation script. This will download the necessary datasets into the `data/01_raw` directory.

```bash
python src/data/make_dataset.py
```

### 5. Launch Jupyter Notebook

You can now explore the analyses in the `notebooks` directory.

```bash
jupyter notebook
```

## Current Notebooks

* **01_pima_diabetes_grid_random_bayesian.ipynb** - Compares Grid Search, Random Search, and Bayesian Optimization on the Pima Indians Diabetes dataset
* **02_breast_cancer_genetic_algorithm.ipynb** - Demonstrates Genetic Algorithm optimization on the Breast Cancer Wisconsin dataset

## Key Findings

Our experiments with different hyperparameter tuning techniques across two datasets yielded several important insights:

### Pima Indians Diabetes Dataset (Grid Search, Random Search, Bayesian Optimization)

1. **Baseline Performance is Competitive**: Surprisingly, the default RandomForestClassifier parameters performed as well as the tuned models on the test set (F1-score: 0.6337), highlighting that sophisticated tuning doesn't always guarantee improvements.

2. **Efficiency Comparison**:
   * Grid Search: Most computationally expensive (115.59s), with highest CV score but no test set advantage
   * Random Search: Significantly faster (36.01s) with identical test performance to baseline
   * Bayesian Optimization (Optuna): Most efficient tuning method (28.49s) with slightly lower test performance

3. **Validation-Test Gap**: All tuning methods found parameters that improved cross-validation scores but didn't generalize equally well to the test set, demonstrating the importance of a proper hold-out test set.

### Breast Cancer Wisconsin Dataset (Genetic Algorithm)

1. **Performance Improvement**: The Genetic Algorithm successfully identified hyperparameters that outperformed the baseline model (F1-score: 0.9585 vs 0.9488, AUC: 0.9917 vs 0.9913).

2. **Efficient Exploration**: The GA converged on a high-performing solution within just 15 generations, efficiently navigating the hyperparameter space.

3. **Best Parameters**: The optimal configuration found was n_estimators=186, max_depth=14, min_samples_split=4.

### Overall Conclusions

* **Start Simple**: Always begin with a baseline model before investing in complex tuning
* **Method Selection**: Choose tuning methods based on computational budget and parameter space complexity
* **Random Search**: Offers excellent balance of performance and efficiency for many problems
* **Genetic Algorithms**: Valuable for complex parameter spaces where simpler methods may be inefficient
* **Bayesian Optimization**: Ideal for projects with limited time budgets requiring efficient search

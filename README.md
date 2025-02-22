# Thesis Project: Time Series Forecasting

## Overview

This repository contains the code and resources for my Master's thesis project on
**Time Series Forecasting**. The project focuses on utilizing machine learning
and statistical models to predict future values in time series datasets.
The code includes data preprocessing, model training and evaluation steps.

## Repository Structure
```
thesis/
│
├── data_processing/           # Scripts for data loading and preprocessing
│   └── preprocess_X.py        # Script for preprocessing X dataset or type of model
│
├── models/                    # Scripts for machine learning models and training
│   └── models_ml.py           # Defines classes of machine learning models
│   └── models_dl.py           # Defines classes of deep learning models
│   └── train_X.py             # Defines functions to train X type of models
│
├── evaluation/                # Scripts for model evaluation and result visualization
│   └── results.py             # Main script for evaluation and results analysis
│
├── utils/                     # Utility functions used across different scripts
│   └── helper.py              # Helper functions for data processing and evaluation
│
├── .gitignore                 # Git ignore file to exclude unnecessary files
├── dl.py                      # Script for running deep learning experiments
├── ml.py                      # Script for running machine learning experiments
└── README.md                  # Project overview and documentation
├── requirements.txt           # Python dependencies for the project
├── stats.py                   # Script for running statistical models experiments
```

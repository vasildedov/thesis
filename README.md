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
├── data_processing/       # Scripts for data loading and preprocessing
│   └── preprocess_X.py    # Script for preprocessing X dataset or type of model
│
├── models/                # Scripts for machine learning models and training
│   └── models_ml.py       # Defines classes of machine learning models
│   └── models_dl.py       # Defines classes of deep learning models
│   └── train_X.py         # Defines functions to train X type of models
│
├── evaluation/            # Scripts for model evaluation and result visualization
│   └── results.py         # Main script for evaluation and results analysis
│
├── utils/                 # Utility functions used across different scripts
│   └── helper.py          # Helper functions for data processing and evaluation
│
├── .gitignore             # Git ignore file to exclude unnecessary files
├── dl.py                  # Script for running deep learning experiments
├── ml.py                  # Script for running machine learning experiments
└── README.md              # Project overview and documentation
├── requirements.txt       # Python dependencies for the project
├── stats.py               # Script for running statistical models experiments
```

## Installation

Follow the steps below to get the project up and running locally.

### 1. Clone the Repository

Start by cloning the repository to your local machine:

```bash
git clone https://github.com/vasildedov/thesis.git
cd thesis
```

### 2. Set Up a Virtual Environment (Optional but Recommended)
It's recommended to use a virtual environment to manage the project's dependencies. If you don't have virtualenv installed, you can install it via pip:

```bash
pip install virtualenv
```
Then, create and activate the virtual environment:
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment (on macOS/Linux)
source venv/bin/activate

# Activate virtual environment (on Windows)
venv\Scripts\activate
```

### 3. Install Dependencies
Use the following command to install the required dependencies listed in the requirements.txt file:

```bash
pip install -r requirements.txt
```

## Usage
Once the dependencies are installed, you can start running the scripts. Below are some usage examples for the key scripts.

### 1. Deep Learning models

```bash
# Run the DL script
python dl.py
```

### 2. Machine Learning models

```bash
# Run the ML script
python ml.py
```

### 3. Statistical models

```bash
# Run the stats script
python stats.py
```

### 4. Evaluating the models
Once the models are trained, use the evaluation script to assess its performance. This script calculates various error metrics (e.g., RMSE, MAE) and exports the results as .tex tables.

```bash
python evaluation/results.py
```

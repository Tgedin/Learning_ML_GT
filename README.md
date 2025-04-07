# Machine Learning Learning Journey

This repository documents my personal journey learning machine learning engineering, complementing my 4Geeks Academy Data Science & Machine Learning bootcamp.

## Project Structure

```
├── README.md                <- Project overview and documentation
├── data/                    <- Data files
│   ├── raw/                 <- Original, immutable data
│   ├── processed/           <- Cleaned, transformed data ready for modeling
│   └── external/            <- Data from third-party sources
├── notebooks/               <- Jupyter notebooks for exploration and analysis
│   └── 01_getting_started.ipynb  <- Introduction notebook
├── models/                  <- Trained models, predictions, and model summaries
├── src/                     <- Source code for use in this project
│   ├── __init__.py          <- Makes src a Python package
│   ├── ml_library_foundations.py  <- ML library foundations
│   ├── utils.py             <- Utility functions
│   ├── data/                <- Scripts for data processing
│   ├── features/            <- Scripts for feature engineering
│   │   └── math_utils.py    <- Mathematical utilities
│   ├── models/              <- Scripts for training models
│   │   └── attention_mechanisms.py  <- Attention mechanisms implementation
│   └── visualization/       <- Scripts for creating visualizations
├── tests/                   <- Test files
│   └── test_setup.py        <- Setup tests
├── configs/                 <- Configuration files
├── docs/                    <- Documentation and references
│   └── Conda & Jupyter Command Reference for Deep Learning.pdf
├── environment.yml          <- Conda environment definition
├── requirements.txt         <- Python package dependencies
└── setup.py                 <- Makes project pip installable
```

## Installation

### Using Conda
```bash
# Create conda environment
conda env create -f environment.yml

# Activate environment
conda activate mldev
```

### Using Pip
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### Development Installation
```bash
# Install project in development mode
pip install -e .
```

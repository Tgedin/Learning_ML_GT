# Only create README.md if it doesn't exist
if [ ! -f README.md ]; then
  cat << EOF > README.md
# Machine Learning Learning Journey

This repository documents my journey learning machine learning engineering, complementing my 4Geeks Academy bootcamp starting April 2025.

## Project Structure

\`\`\`
├── README.md             <- Project overview and documentation
├── data/                 <- Data files
│   ├── raw/              <- Original, immutable data
│   ├── processed/        <- Cleaned, transformed data ready for modeling
│   └── external/         <- Data from third-party sources
├── notebooks/            <- Jupyter notebooks for exploration and analysis
├── models/               <- Trained models, predictions, and model summaries
├── src/                  <- Source code for use in this project
│   ├── data/             <- Scripts for data processing
│   ├── features/         <- Scripts for feature engineering
│   ├── models/           <- Scripts for training models
│   └── visualization/    <- Scripts for creating visualizations
├── configs/              <- Configuration files
└── docs/                 <- Documentation and references
\`\`\`

## Installation

\`\`\`bash
# Create conda environment
conda env create -f environment.yml

# Activate environment
conda activate mldev

# Install development version of the package
pip install -e .
\`\`\`
EOF
fi
# Feature Engineering Utility: Data Preprocessing & Task Generation

A Python utility for streamlined data preprocessing (missing value handling, standardization, feature encoding) and automated train/test task generation for machine learning workflows.

## Overview
This package provides modular functions to:
- Convert column identifiers (letters/integers) to consistent indices
- Impute missing values with multiple strategies
- Standardize numeric features
- Encode categorical features
- Generate train/test splits for multiple target variables
- Orchestrate end-to-end preprocessing via a configuration-driven main function

## Installation
Install required dependencies via pip:
```bash
pip install pandas scikit-learn tqdm xlrd openpyxl
```
- `pandas`: Data manipulation
- `scikit-learn`: Preprocessing and train/test splitting
- `tqdm`: Progress bar visualization
- `openpyxl`: Excel file reading (required for `pd.read_excel`)

## Core Functions

### 1. Column Conversion
```python
convert_to_number_columns(columns: list) -> list[int]
```
Converts column identifiers to 0-based integer indices:
- Accepts **all integers** (e.g., `[0, 1, 2]`) or **all strings** (column letters like `['A', 'B', 'C']` or `['AA', 'AB']`)
- String columns are case-insensitive (e.g., `'a'` → `0`, `'AA'` → `26`)
- Raises `ValueError` for mixed types or non-alphabetic strings

### 2. Missing Value Imputation
```python
fill_missing_data(df: pd.DataFrame, strategy: str, col_names: list[str], **kwargs) -> pd.DataFrame
```
Imputes missing values in specified columns with supported strategies:

| Strategy       | Description                                                                 | Optional kwargs          |
|----------------|-----------------------------------------------------------------------------|--------------------------|
| `mean`         | Fill with column mean                                                       | -                        |
| `median`       | Fill with column median                                                     | -                        |
| `mode`         | Fill with first mode (handles categorical/numeric)                          | -                        |
| `constant`     | Fill with a fixed value                                                     | `value` (default: `0`)   |
| `forward`      | Forward fill (carry last valid value)                                       | `limit` (max fill steps) |
| `backward`     | Backward fill (carry next valid value)                                      | `limit` (max fill steps) |
| `interpolate`  | Interpolate missing values                                                 | `method` (default: `linear`), `limit` |
| `knn`          | K-nearest neighbors imputation (numeric columns only)                       | `n_neighbors` (default: `5`) |
| `drop`         | Drop rows with missing values                                               | `how` (default: `any` → drop if any missing; `all` → drop if all missing) |

### 3. Data Standardization
```python
standardize_data_format(df: pd.DataFrame, strategy: str, col_names: list[str], **kwargs) -> pd.DataFrame
```
Scales numeric features to consistent ranges:

| Strategy       | Description                                                                 |
|----------------|-----------------------------------------------------------------------------|
| `minmax`       | Scale to [0, 1] (MinMaxScaler)                                              |
| `standard`     | Standardize to mean=0, std=1 (StandardScaler)                               |
| `maxabs`       | Scale by maximum absolute value (MaxAbsScaler)                              |
| `robust`       | Robust to outliers (RobustScaler)                                           |

Accepts scikit-learn scaler parameters via `**kwargs` (e.g., `with_mean=False` for `StandardScaler`).

### 4. Feature Encoding
```python
encode_features(df: pd.DataFrame, strategy: str, col_names: list[str], **kwargs) -> pd.DataFrame
```
Encodes categorical features for machine learning:

| Strategy       | Description                                                                 | Optional kwargs                                  |
|----------------|-----------------------------------------------------------------------------|--------------------------------------------------|
| `onehot`       | One-hot encoding (drops original columns)                                   | `prefix`, `prefix_sep` (default: `_`), `dtype` (default: `int`) |
| `labelencode`  | Label encoding (maps categories to 0-n integers)                            | -                                                |
| `ordinal`      | Ordinal encoding (preserves category order)                                 | `categories` (custom category order)             |

### 5. Task Generation
```python
generate_tasks(features_df: pd.DataFrame, targets_df: pd.DataFrame, shuffle: bool, test_size: float, random_state: int, save_tasks_path: str = None, verbose: bool = True) -> list
```
Generates train/test splits for **each target column** (one task per target):
- Returns a list of task dictionaries with `name` (target column), `X_train`, `X_test`, `y_train`, `y_test`
- Saves train/test CSV files to `save_tasks_path` (if provided)
- Uses `train_test_split` under the hood with configurable shuffling and random state

### 6. End-to-End Workflow
```python
feature_engineer(df: pd.DataFrame, config: dict) -> list
```
Orchestrates all preprocessing steps via a single configuration dictionary. Automatically:
1. Selects feature/target columns (via letter/integer identifiers)
2. Applies missing value imputation
3. Standardizes numeric features
4. Encodes categorical features
5. Generates train/test tasks

## Configuration Specification
The `config` dictionary for `feature_engineer` supports the following keys:

| Key                  | Type          | Description                                                                 |
|----------------------|---------------|-----------------------------------------------------------------------------|
| `verbose`            | bool          | Show progress bars (default: `True`)                                        |
| `feature_columns`    | list          | Columns to use as features (letters or integers)                            |
| `target_columns`     | list          | Columns to use as targets (one task per column)                             |
| `fill_missing_data`  | dict          | Keys: imputation strategies; Values: columns to apply strategy to          |
| `standardize_data_format` | dict      | Keys: standardization strategies; Values: columns to apply strategy to     |
| `encode_features`    | dict          | Keys: encoding strategies; Values: columns to apply strategy to            |
| `generate_tasks`     | dict          | Configuration for task generation (see below)                               |

#### `generate_tasks` Sub-Configuration
| Key              | Type          | Description                                                                 |
|------------------|---------------|-----------------------------------------------------------------------------|
| `shuffle`        | bool          | Shuffle data before split (default: `True`)                                 |
| `test_size`      | float         | Proportion of data for test set (default: `0.2`)                            |
| `random_state`   | int           | Random seed for reproducibility (default: `42`)                             |
| `save_tasks_path`| str           | Path to save train/test CSVs (optional)                                     |

## Example Usage
```python
import pandas as pd
from feature_engineer import feature_engineer

# Load raw data (Excel file)
raw_data = pd.read_excel("data/raw_data.xlsx")

# Define preprocessing & task configuration
config = {
    "verbose": True,
    
    # Feature/target columns (letters or integers)
    "feature_columns": ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P'],
    "target_columns": ['Q', 'R', 'S', 'T', 'U'],
    
    # Missing value handling
    "fill_missing_data": {
        "mean": ['E', 'F', 'G', 'H', 'I', 'L', 'N', 'O', 'P'],  # Numeric columns → mean imputation
        "mode": ['A', 'B', 'C', 'D', 'J', 'K', 'M']             # Categorical columns → mode imputation
    },
    
    # Standardization (numeric features)
    "standardize_data_format": {
        "standard": ['E', 'F', 'G', 'H', 'I', 'L', 'N', 'O', 'P']  # Standardize to mean=0, std=1
    },
    
    # Feature encoding (categorical features)
    "encode_features": {
        "onehot": ['A', 'B', 'C', 'D', 'J', 'K', 'M']  # One-hot encode categorical columns
    },
    
    # Task generation
    "generate_tasks": {
        "shuffle": True,
        "test_size": 0.2,
        "save_tasks_path": "./output/tasks/",  # Save CSVs to this directory
        "random_state": 1412                   # Reproducible splits
    }
}

# Run end-to-end preprocessing & task generation
tasks = feature_engineer(raw_data, config)

# Access task data (e.g., first target column)
first_task = tasks[0]
X_train, X_test = first_task["X_train"], first_task["X_test"]
y_train, y_test = first_task["y_train"], first_task["y_test"]
```

## Notes
- Column letters are case-insensitive (e.g., `'a'` → `'A'` → index `0`)
- For `mode` imputation: Uses the first mode if multiple modes exist
- KNN imputation requires numeric columns (standardize first if needed)
- One-hot encoding appends new columns with the format `{prefix}_{category}`
- All functions return copies of the input DataFrame (no in-place modification)

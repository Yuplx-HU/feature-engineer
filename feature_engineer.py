import os
from tqdm import tqdm

import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, LabelEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split


def convert_to_number_columns(columns) -> list[int]:
    if all(isinstance(col, int) for col in columns):
        return columns
    elif all(isinstance(col, str) for col in columns):
        number_columns = []
        for column in columns:
            column = column.upper()
        
            if not column.isalpha():
                raise ValueError("Column letter must contain only alphabetic characters.")
            
            result = 0
            
            for i, char in enumerate(reversed(column)):
                char_value = ord(char) - ord('A') + 1
                result += char_value * (26 ** i)
            
            number_columns.append(result - 1)

        return number_columns
    else:
        raise ValueError("Columns must be all integers or all strings.")


def fill_missing_data(df: pd.DataFrame, strategy: str, col_names: list[str], **kwargs) -> pd.DataFrame:
    df = df.copy()
    
    if not col_names:
        return df
    
    if strategy == "mean":
        fill_values = df[col_names].mean()
        df[col_names] = df[col_names].fillna(fill_values)
            
    elif strategy == "median":
        fill_values = df[col_names].median()
        df[col_names] = df[col_names].fillna(fill_values)
            
    elif strategy == "mode":
        for col in col_names:
            mode_val = df[col].mode()[0] if not df[col].mode().empty else None
            if mode_val is not None:
                df[col] = df[col].fillna(mode_val)
                
    elif strategy == "constant":
        value = kwargs.get('value', 0)
        df[col_names] = df[col_names].fillna(value)
            
    elif strategy == "forward":
        limit = kwargs.get('limit')
        df[col_names] = df[col_names].fillna(method='ffill', limit=limit)
            
    elif strategy == "backward":
        limit = kwargs.get('limit')
        df[col_names] = df[col_names].fillna(method='bfill', limit=limit)
            
    elif strategy == "interpolate":
        method = kwargs.get('method', 'linear')
        limit = kwargs.get('limit')
        df[col_names] = df[col_names].interpolate(method=method, limit=limit, limit_direction='both')
            
    elif strategy == "knn":
        n_neighbors = kwargs.get('n_neighbors', 5)
        imputer = KNNImputer(n_neighbors=n_neighbors)
        imputed_data = imputer.fit_transform(df[col_names])
        df[col_names] = pd.DataFrame(imputed_data, columns=col_names, index=df.index)
        
    elif strategy == "drop":
        how = kwargs.get('how', 'any')
        df = df.dropna(subset=col_names, how=how)
        
    else:
        raise ValueError(f"Unknown fill strategy: {strategy}")
    
    return df


def standardize_data_format(df: pd.DataFrame, strategy: str, col_names: list[str], **kwargs) -> pd.DataFrame:
    df = df.copy()
    
    if not col_names:
        return df
    
    to_scale_cols = col_names
    
    if strategy == "minmax":
        scaler = MinMaxScaler(**kwargs)
    elif strategy == "standard":
        scaler = StandardScaler(**kwargs)
    elif strategy == "maxabs":
        scaler = MaxAbsScaler(**kwargs)
    elif strategy == "robust":
        scaler = RobustScaler(**kwargs)
    else:
        raise ValueError(f"Unknown standardizing strategy: {strategy}")
    
    scaled_data = scaler.fit_transform(df[to_scale_cols])
    df[to_scale_cols] = pd.DataFrame(scaled_data, columns=to_scale_cols, index=df.index)
    
    return df


def encode_features(df: pd.DataFrame, strategy: str, col_names: list[str], **kwargs) -> pd.DataFrame:
    df = df.copy()
    
    if not col_names:
        return df
    
    to_encode_cols = col_names
    
    if strategy == "onehot":
        prefix = kwargs.get('prefix', to_encode_cols)
        prefix_sep = kwargs.get('prefix_sep', '_')
        dtype = kwargs.get('dtype', int)
        
        encoded_df = pd.get_dummies(
            df[to_encode_cols],
            prefix=prefix,
            prefix_sep=prefix_sep,
            columns=to_encode_cols,
            dtype=dtype
        )
        
        df = df.drop(to_encode_cols, axis=1)
        df = pd.concat([df, encoded_df], axis=1)
        
    elif strategy == "labelencode":
        le = LabelEncoder()
        for col in to_encode_cols:
            df[col] = le.fit_transform(df[col].astype(str))
            
    elif strategy == "ordinal":
        ordinal_encoder = OrdinalEncoder(**kwargs)
        encoded_data = ordinal_encoder.fit_transform(df[to_encode_cols])
        df[to_encode_cols] = pd.DataFrame(encoded_data, columns=to_encode_cols, index=df.index)
            
    else:
        raise ValueError(f"Unknown feature strategy: {strategy}")
    
    return df


def generate_tasks(
    features_df: pd.DataFrame,
    targets_df: pd.DataFrame,
    shuffle: bool,
    test_size: float,
    random_state: int,
    save_tasks_path: str = None,
    verbose: bool = True
) -> list:
    if save_tasks_path:
        os.makedirs(save_tasks_path, exist_ok=True)

    feature_columns = features_df.columns.tolist()
    target_columns = targets_df.columns.tolist()
    
    tasks = []
    for target_col in tqdm(target_columns, desc="Generate tasks", disable=not verbose, unit="task"):
        task_df = pd.concat([features_df, targets_df[[target_col]]], axis=1)
        
        train_df, test_df = train_test_split(
            task_df,
            test_size=test_size,
            random_state=random_state,
            shuffle=shuffle
        )
        
        X_train = train_df[feature_columns]
        X_test = test_df[feature_columns]
        y_train = train_df[target_col]
        y_test = test_df[target_col]
        
        tasks.append({
            "name": target_col,
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test
        })
        
        if save_tasks_path:
            train_df.to_csv(os.path.join(save_tasks_path, f"{target_col}_train.csv"), index=False)
            test_df.to_csv(os.path.join(save_tasks_path, f"{target_col}_test.csv"), index=False)
    
    return tasks


def feature_engineer(df: pd.DataFrame, config: dict) -> list:
    features_df = df.iloc[:, convert_to_number_columns(config.get("feature_columns", []))].copy()
    targets_df = df.iloc[:, convert_to_number_columns(config.get("target_columns", []))].copy()
    
    if "fill_missing_data" in config:
        fill_config = config["fill_missing_data"]
        for strategy, columns in fill_config.items():
            features_df = fill_missing_data(features_df, strategy, df.iloc[:, convert_to_number_columns(columns)].columns.tolist())
    
    if "standardize_data_format" in config:
        standardize_config = config["standardize_data_format"]
        for strategy, columns in standardize_config.items():
            features_df = standardize_data_format(features_df, strategy, df.iloc[:, convert_to_number_columns(columns)].columns.tolist())
    
    if "encode_features" in config:
        encode_config = config["encode_features"]
        for strategy, columns in encode_config.items():
            features_df = encode_features(features_df, strategy, df.iloc[:, convert_to_number_columns(columns)].columns.tolist())
    
    tasks_config = config.get("generate_tasks", {})
    tasks = generate_tasks(
        features_df=features_df,
        targets_df=targets_df,
        shuffle=tasks_config.get("shuffle", True),
        test_size=tasks_config.get("test_size", 0.2),
        random_state=tasks_config.get("random_state", 42),
        save_tasks_path=tasks_config.get("save_tasks_path"),
        verbose=config.get("verbose", True)
    )
    
    return tasks

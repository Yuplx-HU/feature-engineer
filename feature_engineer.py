import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import (
    VarianceThreshold,
    SelectKBest, mutual_info_classif, mutual_info_regression,
    RFECV,
)

from npt.utils.model_trainer import create_estimator, create_cv, create_scorer


class OptimalRFE(BaseEstimator, TransformerMixin):
    def __init__(self, task_type: str, estimator: BaseEstimator, random_state: int = random.randint(0, 2**32-1)):
        self.task_type = task_type
        self.estimator = estimator
        self.random_state = random_state
    
    def fit(self, X, y):
        self.rfecv_ = RFECV(
            estimator=self.estimator,
            cv=create_cv(self.task_type, 5, self.random_state),
            scoring=create_scorer(self.task_type),
            min_features_to_select=8,
            n_jobs=-1,
            step=1,
        )
        
        self.rfecv_.fit(X, y)
        
        self.support_ = self.rfecv_.support_
        self.ranking_ = self.rfecv_.ranking_
        self.n_features_ = self.rfecv_.n_features_
        
        if hasattr(X, 'columns'):
            self.feature_names_in_ = np.array(X.columns)
        else:
            self.feature_names_in_ = np.array([f'feature_{i}' for i in range(X.shape[1])])
        
        self.feature_names_out_ = self.feature_names_in_[self.support_]
        self._build_performance_history()
        
        return self
    
    def _build_performance_history(self):
        if hasattr(self.rfecv_, 'cv_results_') and 'mean_test_score' in self.rfecv_.cv_results_:
            scores = self.rfecv_.cv_results_['mean_test_score']
            n_features = self.rfecv_.cv_results_['n_features']
            
            self.performance_df = pd.DataFrame({
                'n_features': n_features,
                'mean_score': scores
            })
        else:
            self.performance_df = None
    
    def transform(self, X):
        if not hasattr(self, 'rfecv_'):
            raise ValueError("OptimalRFE must be fitted before transform")
        return self.rfecv_.transform(X)
    
    def get_support(self):
        return self.support_
    
    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return self.feature_names_out_
        return np.asarray(input_features)[self.support_]
    
    def get_performance_df(self):
        return self.performance_df.copy() if self.performance_df is not None else None
    
    def plot_performance(self, figsize=(10, 6), title="RFE Performance"):
        if self.performance_df is None or self.performance_df.empty:
            return None
        
        fig, ax = plt.subplots(figsize=figsize)
        
        df = self.performance_df
        ax.plot(df['n_features'], df['mean_score'], 'b-o', linewidth=2, markersize=6)
        ax.set_xlabel('Number of Features', fontsize=12)
        ax.set_ylabel('CV Score', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        best_idx = df['mean_score'].idxmax()
        best_features = df.loc[best_idx, 'n_features']
        best_score = df.loc[best_idx, 'mean_score']
        
        ax.scatter(best_features, best_score, color='red', s=80, zorder=5)
        ax.axvline(x=best_features, color='red', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        return fig
    
    def get_best_info(self):
        if self.performance_df is None or self.performance_df.empty:
            return None
        
        best_idx = self.performance_df['mean_score'].idxmax()
        return {
            'best_n_features': int(self.performance_df.loc[best_idx, 'n_features']),
            'best_score': float(self.performance_df.loc[best_idx, 'mean_score']),
            'selected_features': self.feature_names_out_.tolist()
        }


class SelectKWorst(BaseEstimator, TransformerMixin):
    def __init__(self, score_func, k: int):
        self.score_func = score_func
        self.k = k
        
    def fit(self, X, y):
        scores = self.score_func(X, y)
        
        if hasattr(X, 'columns'):
            self.feature_names_in_ = np.array(X.columns)
        else:
            self.feature_names_in_ = np.array([f'feature_{i}' for i in range(X.shape[1])])
        
        k_abs = min(abs(self.k), len(self.feature_names_in_))
        sorted_indices = np.argsort(scores)
        self.selected_indices_ = sorted_indices[:k_abs]
        
        self.support_ = np.zeros(len(self.feature_names_in_), dtype=bool)
        self.support_[self.selected_indices_] = True
        self.feature_names_out_ = self.feature_names_in_[self.support_]
        
        return self
    
    def transform(self, X):
        if not hasattr(self, 'support_'):
            raise ValueError("SelectKWorst must be fitted before transform")
        return X[:, self.support_] if not hasattr(X, 'iloc') else X.iloc[:, self.support_]
    
    def get_support(self):
        return self.support_
    
    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return self.feature_names_out_
        return np.asarray(input_features)[self.support_]


def create_preprocessor(numeric_features, categorical_features, 
                        k_features: int | str = 0, task_type: str = "classification", random_state: int = random.randint(0, 2**32-1)):
    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
    ])
    
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('numeric', numeric_pipeline, numeric_features),
            ('categorical', categorical_pipeline, categorical_features),
        ],
        remainder='drop'
    )
    
    steps = [
        ("preprocessor", preprocessor),
        ("variance_threshold", VarianceThreshold(threshold=0.0)),
    ]
    
    if k_features == "auto":
        steps.append(("feature_selection", OptimalRFE(task_type, create_estimator(task_type, "rf"), random_state)))
    elif isinstance(k_features, int) and k_features != 0:
        score_func = mutual_info_classif if task_type == "classification" else mutual_info_regression
        if k_features > 0:
            selector = SelectKBest(score_func=score_func, k=k_features)
        elif k_features < 0:
            selector = SelectKWorst(score_func=score_func, k=-k_features)
        steps.append(("feature_selection", selector))
    else:
        raise ValueError("`k_features` must be 'auto' or a non-zero integer")
    
    return Pipeline(steps)

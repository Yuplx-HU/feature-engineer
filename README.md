Simple usage:
```Python
preprocessor = create_preprocessor(numeric_feature_names, categorical_feature_names,
                                   k_features=k_features, task_type=task_type, random_state=random_state)
X = preprocessor.transform(X)
// or
X = preprocessor.fit_transform(X, y)
```

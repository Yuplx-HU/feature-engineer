Simple usage:
```Python
preprocessor = create_preprocessor(numeric_feature_names, categorical_feature_names,
                                   task_type=task_type, k_features=k_features, random_state=random_state)
X = preprocessor.transform(X)
# OR
X = preprocessor.fit_transform(X, y)
```

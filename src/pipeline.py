from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


def create_pipeline(numeric_features, categorical_features):
    numeric_preprocessing = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )

    categorical_preprocessing = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("label_encoder", OneHotEncoder()),
        ]
    )

    preprocessing = ColumnTransformer(
        [
            ("numeric", numeric_preprocessing, numeric_features),
            ("categorical", categorical_preprocessing, categorical_features),
        ]
    )

    param_grid = [
        {
            "model": [RandomForestClassifier()],
            "model__n_estimators": [50, 100, 200],
            "model__max_depth": [5, 10, None],
            "model__criterion": ["gini", "entropy", "log_loss"],
        },
        {"model": [LogisticRegression(max_iter=1000)], "model__C": [0.1, 1.0, 10.0]},
        {
            "model": [SVC()],
            "model__C": [0.1, 1.0, 10.0],
            "model__kernel": ["rbf", "linear", "poly"],
            "model__degree": [3, 4, 5],
        },
        {
            "model": [GradientBoostingClassifier()],
            "model__learning_rate": [0.01, 0.1, 1],
            "model__n_estimators": [50, 100, 200],
            "model__min_samples_split": [2, 3, 5],
            "model__max_depth": [2, 3, 4, 5],
        },
    ]

    pipeline = Pipeline([("preprocessing", preprocessing), ("model", None)])

    gs = GridSearchCV(
        estimator=pipeline, cv=5, n_jobs=-1, scoring="recall", param_grid=param_grid
    )

    return gs

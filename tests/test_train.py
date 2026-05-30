from src.data_loader import load_data
from src.pipeline import create_pipeline
from src.train import train_model
from src.predict import model_predict
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score

def test_train():
    df = load_data('data/heart.csv')

    X = df.drop(columns=['HeartDisease'])
    y = df['HeartDisease']

    categorial_features = X.select_dtypes(['object']).columns
    numeric_features = X.select_dtypes(['number']).columns
    pipeline = create_pipeline(categorical_features=categorial_features, numeric_features=numeric_features)

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2)
    train_model(pipeline, X_train, y_train)
    y_pred = model_predict(pipeline, X_test)

    assert recall_score(y_test, y_pred) > 0.85

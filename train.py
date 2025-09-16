from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from preprocess import load_and_preprocess

def train_model(data_path):
    X_train, X_test, y_train, y_test = load_and_preprocess(data_path)

    # Train Random Forest (you can switch model here)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Accuracy
    acc = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {acc:.2f}")

    return model, X_test, y_test

if __name__ == "__main__":
    model, X_test, y_test = train_model("data/crop_recommendation.csv")

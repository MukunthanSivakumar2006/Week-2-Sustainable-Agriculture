from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from train import train_model

def evaluate(data_path):
    model, X_test, y_test = train_model(data_path)

    # Predictions
    y_pred = model.predict(X_test)

    # Classification Report
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

if __name__ == "__main__":
    evaluate("data/crop_recommendation.csv")

from scripts.preprocess import load_and_preprocess
from scripts.train_model import train_model
from scripts.evaluate import evaluate_model
from tensorflow.keras.models import load_model  # This import is okay if you plan to use it later

def main():
    print("Loading and preprocessing data...")
    X_train, X_test, y_train, y_test, features = load_and_preprocess("data/ckd.csv")

    print("Training model...")
    model, _ = train_model(X_train, y_train, X_test, y_test)

    print("Evaluating model...")
    evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    main()


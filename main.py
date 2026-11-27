"""CLI entry point for the house price pipeline."""

import argparse

from model_pipeline import TARGET, evaluate_model, load_model, prepare_data, save_model, train_model


def main():
    parser = argparse.ArgumentParser(description="House price MLOps pipeline")
    parser.add_argument(
        "--step", choices=["prepare", "train", "evaluate", "predict"], required=True
    )
    parser.add_argument("--data", default="data/houses.csv")
    parser.add_argument("--target", default=TARGET)
    parser.add_argument("--model-path", default="model.joblib")
    args = parser.parse_args()

    X_train, X_test, y_train, y_test = prepare_data(args.data, args.target)

    if args.step == "prepare":
        print(f"train: {X_train.shape}, test: {X_test.shape}")
    elif args.step == "train":
        model = train_model(X_train, y_train)
        save_model(model, args.model_path)
        print("Model saved to", args.model_path)
        print(evaluate_model(model, X_test, y_test))
    elif args.step == "evaluate":
        model = load_model(args.model_path)
        print(evaluate_model(model, X_test, y_test))
    elif args.step == "predict":
        model = load_model(args.model_path)
        print(model.predict(X_test.head(5)))


if __name__ == "__main__":
    main()

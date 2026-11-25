"""CLI entry point for the house price pipeline."""
import argparse

from model_pipeline import (TARGET, evaluate_model, load_model, prepare_data,
                            save_model, train_model)


def main():
    parser = argparse.ArgumentParser(description="House price MLOps pipeline")
    parser.add_argument("--step", choices=["train", "evaluate"], required=True)
    parser.add_argument("--data", default="data/houses.csv")
    parser.add_argument("--target", default=TARGET)
    parser.add_argument("--model-path", default="model.joblib")
    args = parser.parse_args()

    X_train, X_test, y_train, y_test = prepare_data(args.data, args.target)

    if args.step == "train":
        model = train_model(X_train, y_train)
        save_model(model, args.model_path)
        print("Model saved to", args.model_path)
    else:
        model = load_model(args.model_path)
    print(evaluate_model(model, X_test, y_test))


if __name__ == "__main__":
    main()

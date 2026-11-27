"""Registers one deployment per flow and serves them (the 'worker')."""

from prefect import serve

from pipeline_prefect import (
    all_flow,
    api_flow,
    clean_flow,
    code_flow,
    evaluate_flow,
    install_flow,
    predict_flow,
    prepare_flow,
    test_flow,
    train_flow,
)

if __name__ == "__main__":
    serve(
        all_flow.to_deployment(
            name="ml-pipeline-all", cron="0 2 * * *", tags=["mlops", "full-pipeline"]
        ),
        install_flow.to_deployment(name="ml-pipeline-install", tags=["mlops", "setup"]),
        code_flow.to_deployment(name="ml-pipeline-code", tags=["mlops", "code-quality"]),
        prepare_flow.to_deployment(name="ml-pipeline-prepare", tags=["mlops", "data"]),
        train_flow.to_deployment(name="ml-pipeline-train", tags=["mlops", "training"]),
        evaluate_flow.to_deployment(name="ml-pipeline-evaluate", tags=["mlops", "evaluation"]),
        predict_flow.to_deployment(name="ml-pipeline-predict", tags=["mlops", "prediction"]),
        test_flow.to_deployment(name="ml-pipeline-test", tags=["mlops", "tests"]),
        clean_flow.to_deployment(name="ml-pipeline-clean", tags=["mlops", "maintenance"]),
        api_flow.to_deployment(name="ml-pipeline-api", tags=["mlops", "serving"]),
    )

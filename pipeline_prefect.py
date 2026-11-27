"""Workshop 3 - ML pipeline with Prefect."""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from prefect import flow, get_run_logger, task

ROOT = Path(__file__).resolve().parent
VENV = ROOT / "venv"
MAX_LINE = 100


def py():
    venv_py = VENV / "bin" / "python"
    return str(venv_py) if venv_py.exists() else sys.executable


def sh(cmd, check=True):
    logger = get_run_logger()
    logger.info("$ %s", " ".join(cmd))
    res = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if res.stdout.strip():
        logger.info(res.stdout.strip())
    if res.stderr.strip():
        logger.warning(res.stderr.strip())
    if check and res.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")
    return res


# ------------------------------ TASKS ------------------------------
@task(name="create-venv")
def create_venv_task():
    if not (VENV / "bin" / "python").exists():
        sh([sys.executable, "-m", "venv", str(VENV)])


@task(name="install-requirements")
def install_task():
    sh([py(), "-m", "pip", "install", "-r", "requirements.txt"])


@task(name="format-code")
def format_task():
    sh([py(), "-m", "black", ".", "--line-length", str(MAX_LINE), "--extend-exclude", "venv"])


@task(name="quality-check")
def quality_task():
    # non-blocking: warnings are logged but do not stop the pipeline
    sh(
        [
            py(),
            "-m",
            "flake8",
            ".",
            f"--max-line-length={MAX_LINE}",
            "--exclude=venv,__pycache__,mlruns",
        ],
        check=False,
    )


@task(name="security-check")
def security_task():
    sh([py(), "-m", "bandit", "-r", ".", "-x", "./venv,./tests", "-ll"], check=False)


@task(name="create-sample-test")
def sample_test_task():
    Path(ROOT / "tests").mkdir(exist_ok=True)
    sample = ROOT / "tests" / "test_sample.py"
    if not sample.exists():
        sample.write_text(
            "from model_pipeline import DROP_COLS, prepare_data\n\n\n"
            "def test_prepare_data_drops_leaky_columns():\n"
            "    X_train, X_test, _, _ = prepare_data('data/houses.csv')\n"
            "    for col in DROP_COLS:\n"
            "        assert col not in X_train.columns\n"
            "    assert len(X_train) > len(X_test)\n"
        )


@task(name="run-unit-tests")
def run_tests_task():
    sh([py(), "-m", "pytest", "tests", "-v"])


@task(name="prepare-data")
def prepare_task():
    sh([py(), "main.py", "--step", "prepare"])


@task(name="train-model")
def train_task():
    sh([py(), "main.py", "--step", "train"])


@task(name="evaluate-model")
def evaluate_task():
    sh([py(), "main.py", "--step", "evaluate"])


@task(name="predict")
def predict_task():
    sh([py(), "main.py", "--step", "predict"])


@task(name="clean-caches")
def clean_task():
    for d in ROOT.rglob("__pycache__"):
        if "venv" not in d.parts:
            shutil.rmtree(d, ignore_errors=True)
    shutil.rmtree(ROOT / ".pytest_cache", ignore_errors=True)


@task(name="launch-api")
def api_task():
    get_run_logger().info("API on http://127.0.0.1:8000/docs (Ctrl+C to stop)")
    subprocess.run(
        [py(), "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"], cwd=ROOT
    )


# ------------------------------ FLOWS ------------------------------
@flow(name="install", log_prints=True)
def install_flow():
    create_venv_task()
    install_task()


@flow(name="code", log_prints=True)
def code_flow():
    format_task()
    quality_task()
    security_task()
    sample_test_task()
    run_tests_task()


@flow(name="test", log_prints=True)
def test_flow():
    sample_test_task()
    run_tests_task()


@flow(name="prepare", log_prints=True)
def prepare_flow():
    prepare_task()


@flow(name="train", log_prints=True)
def train_flow():
    train_task()


@flow(name="evaluate", log_prints=True)
def evaluate_flow():
    evaluate_task()


@flow(name="predict", log_prints=True)
def predict_flow():
    predict_task()


@flow(name="clean", log_prints=True)
def clean_flow():
    clean_task()


@flow(name="api", log_prints=True)
def api_flow():
    api_task()


@flow(name="all", log_prints=True)
def all_flow():
    install_flow()
    code_flow()
    prepare_flow()
    train_flow()
    evaluate_flow()
    predict_flow()


FLOWS = {
    "all": all_flow,
    "install": install_flow,
    "code": code_flow,
    "prepare": prepare_flow,
    "train": train_flow,
    "evaluate": evaluate_flow,
    "predict": predict_flow,
    "test": test_flow,
    "clean": clean_flow,
    "api": api_flow,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--flow", choices=FLOWS.keys(), default="all")
    args = parser.parse_args()
    FLOWS[args.flow]()

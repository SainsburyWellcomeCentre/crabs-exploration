import os

# Allow MLflow's filesystem tracking backend (deprecated since MLflow 3.8, but
# still required for our file:// tracking URIs and file-store checkpoints). Set
# here, before any submodule imports mlflow, so it applies package-wide. Set as
# a default so it can still be overridden via the environment.
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

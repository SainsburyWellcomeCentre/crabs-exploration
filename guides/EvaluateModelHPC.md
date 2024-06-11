# Evaluate a trained detector model in the cluster

1.  **Preparatory steps**

    - If you are not connected to the SWC network: connect to the SWC VPN.

1.  **Connect to the SWC HPC cluster**

    ```
    ssh <SWC-USERNAME>@ssh.swc.ucl.ac.uk
    ssh hpc-gw1
    ```

    It may ask for your password twice. To set up SSH keys for the SWC cluster, see [this guide](https://howto.neuroinformatics.dev/programming/SSH-SWC-cluster.html#ssh-keys).

1.  **Download the training script from the 🦀 repository**

    To do so, run any of the following commands. They will download a bash script for training (`run_evaluate_single.sh` or `run_evaluate_array.sh`) to the current working directory.

    The download the version of these files in the `main` branch of the [🦀 repository](https://github.com/SainsburyWellcomeCentre/crabs-exploration), run one of the following commands.

    - To train a single job: download the `run_evaluate_single.sh` file

      ```
      curl https://raw.githubusercontent.com/SainsburyWellcomeCentre/crabs-exploration/main/bash_scripts/run_training_single.sh > run_training_single.sh
      ```

    - To train an array job: download the `run_evaluate_array.sh` file
      ```
      curl https://raw.githubusercontent.com/SainsburyWellcomeCentre/crabs-exploration/main/bash_scripts/run_training_array.sh > run_training_array.sh
      ```

    These bash scripts will launch a SLURM job that:

    - gets the 🦀 package from git,
    - installs it in the compute node,
    - and runs a training job.

> [!TIP]
> To retrieve a version of these files that is different from the files at the tip of `main`, edit the remote file path in the curl command:
>
> - For example, to download the version of the file at the tip of a branch called `<BRANCH-NAME>`, edit the path above to replace `main` with `<BRANCH-NAME>`:
>   ```
>   https://raw.githubusercontent.com/SainsburyWellcomeCentre/crabs-exploration/<BRANCH-NAME>/bash_scripts/run_training_single.sh
>   ```
> - To download the version of the file of a specific commit, replace `main` with `blob/<COMMIT-HASH>`:
>   ```
>   https://raw.githubusercontent.com/SainsburyWellcomeCentre/crabs-exploration/blob/<COMMIT-HASH>/bash_scripts/run_training_single.sh
>   ```

4.  **Edit the bash script!**

    Unlike the training, during evaluation, we need to ensure the correct trained model and seed number is used. All the parameters used in any training is logged into `mlflow`.

    We can see the perfomance of each training session by inspecting the `metrics` tab in `mlflow UI` where the `training loss`, `validation precision` and `validation recall` are plotted. The `seed number` and the `checkpoint path` are logged in `parameters` section under `overview` tab.

    When launching a single or an array job, we may want to edit in the bash script:

    - The `EXPERIMENT_NAME`. This should roughly reflect the reason for running the evaluation (for example, include `data_augm_evaluation` for a data augmentation ablation study). Otherwise we use the name of the dataset (e.g., Sep2023_evaluation).
    - The `MLFLOW_FOLDER`. By default, we point to the "scratch" folder at `/ceph/zoo/users/sminano/ml-runs-all/ml-runs-scratch` . This folder holds runs that we don't need to keep. For runs we would like to consider for evaluation, we will instead point to the folder at `/ceph/zoo/users/sminano/ml-runs-all/ml-runs`. Make sure the `SPLIT_SEED` is the same one used during training.

    Less frequently, one may need to edit:

    - the `DATASET_DIR`: usually we train on the "Sep2023_labelled" dataset;
    - the `CONFIG_FILE`: usually we point to the same file we used to train the model at `/ceph/zoo/users/sminano/cluster_train_config.yaml` which we can edit. Note that all config parameters are logged in MLflow, so we don't need to keep track of that file;
    - the `GIT_BRANCH`, if we want to use a specific version of the 🦀 package. Usually we will run the version of the 🦀 package in `main`.

    Additionally for an array job, one may want to edit the number of jobs in the array (by default set to 3):

    - this would mean editing the line that start with `#SBATCH --array=0-n%m` in the `run_evaluation_array.sh` script. That command specifies to run `n` separate jobs, but not more than `m` at a time.
    - if the number of jobs in the array is edited, the variable `LIST_SEEDS` needs to be modified accordingly, otherwise we will get an error when launching the job.

5.  **Optional CLI arguments**

    One may add additional CLI argument in the bash script.

    - `save_frames` might be useful if we want to save the prediction of every frame for further analysis.

6.  **Run the evaluation job using the SLURM scheduler**

    To launch a job, use the `sbatch` command with the relevant training script:

    ```
    sbatch <path-to-training-bash-script>
    ```

7.  **Check the status of the training job**

    To do this, we can:

    - Check the SLURM logs: these should be created automatically in the directory from which the `sbatch` command is run.
    - Run supporting SLURM commands (see [below](#some-useful-slurm-commands)).
    - Check the MLFlow logs. To do this, first create or activate an existing conda environment with `mlflow` installed, and then run the `mlflow` command from the login node.

      - Create and activate a conda environment.
        ```
        module load miniconda
        conda create -n mlflow-env python=3.10 mlflow -y
        conda activate mlflow-env
        ```
      - Run `mlflow` to visualise the results logged to the `ml-runs` folder.

        - If using the "scratch" folder:

          ```
          mlflow ui --backend-store-uri file:////ceph/zoo/users/sminano/ml-runs-all/ml-runs-scratch
          ```

        - If using the selected runs folder:

          ```
          mlflow ui --backend-store-uri file:////ceph/zoo/users/sminano/ml-runs-all/ml-runs
          ```

### Some useful SLURM commands

To check the status of your jobs in the queue

```
squeue -u <username>
```

To show details of the latest jobs (including completed or cancelled jobs)

```
sacct -X -u <username>
```

To specify columns to display use `--format` (e.g., `Elapsed`)

```
sacct -X --format="JobID, JobName, Partition, Account, State, Elapsed" -u <username>
```

To check specific jobs by ID

```
sacct -X -j 3813494,3813184
```

To check the time limit of the jobs submitted by a user (for example, `sminano`)

```
squeue -u sminano --format="%i %P %j %u %T %l %C %S"
```

To cancel a job

```
scancel <jobID>
```

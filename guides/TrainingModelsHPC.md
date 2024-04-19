# Train a detector in the cluster

0.  **Prep**

    - If not in the SWC network: connect to the SWC VPN.

1.  **Connect to the SWC HPC cluster**

    ```
    ssh hpc-gw1
    ```

    It may ask for your password twice. To set up SSH keys for the SWC cluster, see [this guide](https://howto.neuroinformatics.dev/programming/SSH-SWC-cluster.html#ssh-keys).

1.  **Fetch the training script from the 🦀 repository**

    To do so, run any of the following commands. They will place the bash script for training in the current working directory. The files retrieved will be the versions in the `main` branch of the repository.

    - To train a single job: fetch the `run_training_single.sh` file

      ```
      curl https://raw.githubusercontent.com/SainsburyWellcomeCentre/crabs-exploration/main/bash_scripts/run_training_single.sh > run_training_single.sh
      ```

    - To train an array job: fetch the `run_training_array.sh` file
      ```
      curl https://raw.githubusercontent.com/SainsburyWellcomeCentre/crabs-exploration/main/bash_scripts/run_training_array.sh > run_training_array.sh
      ```

    To retrieve a different version of any of these files, from the one at the tip of `main`:

    - replace `main` in the remote file path after `curl` with the relevant branch, or
    - replace `main` with `blob/commit-hash` to retrieve a specific commit.

    These bash scripts fetch the crabs package from git, install it in the compute node, and run a training job.

1.  **Edit the training bash script if required**

    Ideally, we won't edit the training scripts much. If we find we do, then we may want to consider moving the relevant parameters to the config file.

    When launching a single or an array job, one may want to edit in the bash script:

    - The `--experiment_name` (line 111). This should roughly reflect the reason for running the experiment (for example, include `data_augm` for a data augmentation ablation study). Otherwise we use the name of the dataset (Sep2023).
    - The `mlflow_folder` (line 113). By default, we point to the "scratch" folder. This holds runs that we don't need to keep. For runs we would like to consider for evaluation, we will instead point to the folder at `/ceph/zoo/users/sminano/ml-runs-all/ml-runs`.

    Less frequently, one may need to edit:

    - the `DATASET_DIR`: usually we train on the "Sep2023_labelled" dataset;
    - the `TRAIN_CONFIG_FILE`: usually we point to the file at `/ceph/zoo/users/sminano/cluster_train_config.yaml` which we can edit. Note that all config parameters are logged in MLflow;
    - the `GIT_BRANCH`, if we want to use a specific version of the package. Usually we will run the version in `main`.

    Additionally for an array job, one may want to edit the number of jobs in the array (by default set to 3):

    - if the number of jobs in the array is edited, the `LIST_SEEDS` needs to be modified accordingly, otherwise we will get an error when launching the job.

1.  Edit the config if required

    The config file the bash scripts point to is located at:

    ```
    /ceph/zoo/users/sminano/cluster_train_config.yaml
    ```

    Ideally, we edit the config file more than the bash scripts. All the info in the config file is anyways recorded for each run in its MLflow logs.

    > [!CAUTION]
    >
    > If we launch a job and then modify the config file before the job has been able to read it, we may be using an undesired version of the config in our job. To avoid this, it is best to wait until you can verify in MLflow that the job has the expected config parameters, and then edit the file for a new job if needed.

    Some parameters we often modify in the config:

    - `num_epochs`
    - `batch_size_train`
    - `batch_size_test`
    - `batch_size_val`
    - `checkpoint_saving` parameters
    - `iou_threshold`
    - `num_workers`

1.  **Run the training job using the SLURM scheduler**

    To launch a job, use the `sbatch` command with the training script you fetched (and maybe edited):

    ```
    sbatch <path-to-fetched-training-bash-script>
    ```

1.  **Check the status of the training job**

    To do this, we can:

    - Check the SLURM logs: these should be created automatically in the directory from which the `sbatch` command is run.
    - Run supporting SLURM commands (see [below](#some-useful-slurm-commands)).
    - Check the MLFlow logs. To do this, first create or activate an existing conda environment with `mlflow` installed, and then run the `mlflow` command from the login node.

      - Activate the conda environment.
        ```
        conda activate mlflow-env
        ```
      - Run `mlflow` to visualise the results in the ml-runs folder.

        If using the "scratch" folder:

        ```
        mlflow ui --backend-store-uri file:////ceph/zoo/users/sminano/ml-runs-all/ml-runs-scratch
        ```

        If using the selected runs folder:

        ```
        mlflow ui --backend-store-uri file:////ceph/zoo/users/sminano/ml-runs-all/ml-runs
        ```

       <details>
          <summary>See how to create the mlflow conda environment</summary>

             - Load miniconda
                ```
                module load miniconda
                ```

             - Create a conda environment and install mlflow
                ```
                conda create -n mlflow-env python=3.10 mlflow -y
                ```

       </details>

### Some useful SLURM commands

To check the status of your jobs in the queue

```
squeue -u sminano
```

To show details of the latest jobs (including completed or cancelled jobs)

```
sacct -X -u sminano
```

To specify columns to display use `--format` (e.g., `Elapsed`)

```
sacct -X --format="JobID, JobName, Partition, Account, State, Elapsed" -u sminano
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

### Troubleshooting

...

# Train a detector in the cluster

1.  **Preparatory steps**

    - If you are not connected to the SWC network: connect to the SWC VPN.

1.  **Connect to the SWC HPC cluster**

    ```
    ssh <SWC-USERNAME>@ssh.swc.ucl.ac.uk
    ssh hpc-gw1
    ```

    It may ask for your password twice. To set up SSH keys for the SWC cluster, see [this guide](https://howto.neuroinformatics.dev/programming/SSH-SWC-cluster.html#ssh-keys).

1.  **Download the training script from the 🦀 repository**

    To do so, run any of the following commands. They will download a bash script for training (`run_training_single.sh` or `run_training_array.sh`) to the current working directory.

    The download the version of these files in the `main` branch of the [🦀 repository](https://github.com/SainsburyWellcomeCentre/crabs-exploration), run one of the following commands.

    - To train a single job: download the `run_training_single.sh` file

      ```
      curl https://raw.githubusercontent.com/SainsburyWellcomeCentre/crabs-exploration/main/bash_scripts/run_training_single.sh > run_training_single.sh
      ```

    - To train an array job: download the `run_training_array.sh` file
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

4.  **Edit the training bash script if required**

    Ideally, we won't make major edits to the training scripts. If we find we do, then we may want to consider moving the relevant parameters to the config file, or making them a CLI argument.

    When launching a single or an array job, we may want to edit in the bash script:

    - The `EXPERIMENT_NAME`. This should roughly reflect the reason for running the experiment (for example, include `data_augm` for a data augmentation ablation study). Otherwise we use the name of the dataset (e.g., Sep2023).
    - The `MLFLOW_FOLDER`. By default, we point to the "scratch" folder at `/ceph/zoo/users/sminano/ml-runs-all/ml-runs-scratch` . This folder holds runs that we don't need to keep. For runs we would like to consider for evaluation, we will instead point to the folder at `/ceph/zoo/users/sminano/ml-runs-all/ml-runs`.

    Less frequently, one may need to edit:

    - the `DATASET_DIR`: usually we train on the "Sep2023_labelled" dataset;
    - the `TRAIN_CONFIG_FILE`: usually we point to the file at `/ceph/zoo/users/sminano/cluster_train_config.yaml` which we can edit. Note that all config parameters are logged in MLflow, so we don't need to keep track of that file;
    - the `GIT_BRANCH`, if we want to use a specific version of the 🦀 package. Usually we will run the version of the 🦀 package in `main`.

    Additionally for an array job, one may want to edit the number of jobs in the array (by default set to 3):

    - this would mean editing the line that start with `#SBATCH --array=0-n%m` in the `run_training_array.sh` script. That command specifies to run `n` separate jobs, but not more than `m` at a time.
    - if the number of jobs in the array is edited, the variable `LIST_SEEDS` needs to be modified accordingly, otherwise we will get an error when launching the job.

1.  **Edit the config YAML file if required**

    The config file the bash scripts point to is located at:

    ```
    /ceph/zoo/users/sminano/cluster_train_config.yaml
    ```

    Ideally, we edit the config file more often than we edit the bash scripts. All the info in the config file is recorded for each run in MLflow.

    Some parameters we often modify in the config:

    - `num_epochs`
    - `batch_size_train`
    - `batch_size_test`
    - `batch_size_val`
    - `checkpoint_saving` parameters
    - `iou_threshold`
    - `num_workers`

> [!CAUTION]
>
> If we launch a job and then modify the config file _before_ the job has been able to read it, we may be using an undesired version of the config in our job! To avoid this, it is best to wait until you can verify in MLflow that the job has the expected config parameters (and then edit the file to launch a new job if needed).

6. **Checkpoint**

   The `checkpoint_path` argument can be useful. There are two primary options related to checkpoints:

   - Resume training

     - This option is useful for interrupted training sessions or extending training duration.
     - If training is disrupted and stops mid-way, you can resume it by adding `--checkpoint_path $CKPT_PATH \` to your bash script.
     - The training will pick up from the last saved epoch and continue until the specified n_epoch.
     - Similarly, if training completes but you want to extend it based on metric evaluations, you can increase the n_epoch value (e.g., from `n` to `n + y`). Again, use `--checkpoint_path $CKPT_PATH \` in your bash script, and training will resume from epoch `n` to `n + y`.
     - Ensure the `save_weights_only` parameter under `checkpoint_saving` in the config file is set to `False` to resume training, as this option requires loading both weights and the training state.

   - Fine-tunning
     - This option is useful for fine-tuning a pre-trained model on a different dataset.
     - It loads the weights from a checkpoint, allowing you to leverage pre-trained weights from another dataset.
     - Add `--checkpoint_path $CKPT_PATH \` to your bash script to use this option.
     - Set the `save_weights_only` parameter under `checkpoint_saving` in the config file to `True`, as only the weights are needed for fine-tuning.

7. **Optional argument - Optuna**

   We have the option to run [Optuna](https://optuna.org) which is a hyperparameter optimization framework that allows us the find the best hyperparameters for our model.

   Currently, we can use Optuna to find the optimal number of epochs and the optimal learning rate, within a specified range of values. These will be optimal by maximizing the validation precision and recall. We then use these optimal parameters to train the model.

   To run an `Optuna` hyperparameter optimization, we need to specify the range of values we wish to optimize in the configuration file (`/crabs/detection_tracking/config/faster_rcnn.yaml`). Under `optuna`, specify the following:

   - `n_trials`: The number of trials you want Optuna to run. Each trial will explore a different combination of hyperparameters within the defined search space., and their performance metrics will be compared.
   - `learning_rate`: The lower bound and the upper bound of the learning rate parameter to consider.
   - `n_epochs`: The lower bound and the upper bound of the number of epochs to consider.

   To run Optuna, we can add the `--optuna` argument to the bash script. For example:

   ```
   train-detector  \
   --dataset_dirs $DATASET_DIR \
   --config_file $TRAIN_CONFIG_FILE \
   --accelerator gpu \
   --experiment_name $EXPERIMENT_NAME \
   --seed_n $SPLIT_SEED \
   --mlflow_folder $MLFLOW_FOLDER \
   --optuna
   ```

8. **Run the training job using the SLURM scheduler**

   To launch a job, use the `sbatch` command with the relevant training script:

   ```
   sbatch <path-to-training-bash-script>
   ```

9. **Check the status of the training job**

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

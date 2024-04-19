# Train a detector in the cluster

0. If not in the SWC network: connect to the SWC VPN

1. Connect to the SWC HPC

   ```
   ssh hpc-gw1
   ```

   It may ask for your password twice. To set up SSH keys for the SWC cluster, see [this guide](https://howto.neuroinformatics.dev/programming/SSH-SWC-cluster.html#ssh-keys)

2. Fetch the bash script for training from the repository

   To do so, run any of the following commands. They will place the bash script for training in the current working directory. The files retrieved will be the versions in the `main` branch of the repository.

   To train a single job: fetch the `run_training_single.sh` file

   ```
   curl https://raw.githubusercontent.com/SainsburyWellcomeCentre/crabs-exploration/main/bash_scripts/run_training_single.sh > run_training_single.sh
   ```

   To train an array job: fetch the `run_training_array.sh` file

   ```
   curl https://raw.githubusercontent.com/SainsburyWellcomeCentre/crabs-exploration/main/bash_scripts/run_training_array.sh > run_training_array.sh
   ```

   To retrieve a different versions of these files, from the ones at the tip of `main`:

   - replace `main` in the remote file path with the relevant branch, or
   - replace `main` with `blob/commit-hash` to retrieve a specific commit.

   These bash scripts fetch the crabs package from git, install it in the compute node, and run a training job.

3. Edit the bash script if required.

   Ideally, we don't edit the bash scripts much for training. If we find we do, then we may want to consider moving the relevant parameters to the config file.

   For launching a single or an array job, some things one may want to edit in the bash script are:

   - The `--experiment_name` (line 111). This may reflect the reason for running an experiment, for example, a data augmentation ablation study. Otherwise we use the name of the dataset (Sep2023).
   - The `mlflow_folder` (line 113). By default, we point to the "scratch" folder. For runs we would like to consider for evaluation, we may want to point to the selected runs folder at `/ceph/zoo/users/sminano/ml-runs-all/ml-runs`.

   Less frequently, one may need to edit:

   - the `DATASET_DIR`: usually we train on the "Sep2023_labelled" dataset;
   - the `TRAIN_CONFIG_FILE`: usually we point to the file at `/ceph/zoo/users/sminano/cluster_train_config.yaml` which we can edit. Note that all config parameters are logged in MLflow;
   - the `GIT_BRANCH`, if we want to use a specific version of the package. Usually we will run the version in `main`.

   Additionally for an array job, one may want to edit the number of jobs in the array (by default set to 3):

   - if the number of jobs in the array is edited, the `LIST_SEEDS` needs to be modified accordingly, otherwise we will get an error when launching the job.

4. Edit the config if required

   The config file the bash scripts point to is located at `/ceph/zoo/users/sminano/cluster_train_config.yaml`.

   Ideally we edit the config file more than the bash scripts. All the info in the config file is recorded for each run in MLflow logs.

   ATTENTION! If we launch a job and then modify the config file before the job has been able to read it, we may be using an undesired version of the config in our job. To avoid this, it is best to wait until you can verify in MLflow the launched has the expected config parameters

   ...

   Some parameters one may modify

   ....

5. Run the training job using the SLURM scheduler

   ```
   sbatch <path-to-fetched-bash-script>
   ```

6. To check the status of a training job you can:

   - Check the SLURM logs.
     The
   - Run supporting SLURM commands (see [below](#some-useful-slurm-commands)).
   - Check the MLFlow logs. To do this, run:

     ```

     ```

## Some useful SLURM commands

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

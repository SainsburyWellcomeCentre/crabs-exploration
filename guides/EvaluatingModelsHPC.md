# Evaluate a list of detectors in the cluster

1.  **Preparatory steps**

    - If you are not connected to the SWC network: connect to the SWC VPN.

2.  **Connect to the SWC HPC cluster**

    ```
    ssh <SWC-USERNAME>@ssh.swc.ucl.ac.uk
    ssh hpc-gw1
    ```

    It may ask for your password twice. To set up SSH keys for the SWC cluster, see [this guide](https://howto.neuroinformatics.dev/programming/SSH-SWC-cluster.html#ssh-keys).

3.  **Download the evaluation script from the 🦀 repository**

    To do so, run the following command, which will download a bash script called `run_evaluation_array.sh` to the current working directory.
    ```
    curl https://raw.githubusercontent.com/SainsburyWellcomeCentre/crabs-exploration/main/bash_scripts/run_evaluation_array.sh > run_evaluation_array.sh
    ```

    This bash script launches a SLURM array job that evaluates an array of trained models. The version of the bash script downloaded is the one at the tip of the `main` branch in the [🦀 repository](https://github.com/SainsburyWellcomeCentre/crabs-exploration).


> [!TIP]
> To retrieve a version of these files that is different from the files at the tip of `main`, edit the remote file path in the curl command:
>
> - For example, to download the version of the file at the tip of a branch called `<BRANCH-NAME>`, edit the path above to replace `main` with `<BRANCH-NAME>`:
>   ```
>   https://raw.githubusercontent.com/SainsburyWellcomeCentre/crabs-exploration/<BRANCH-NAME>/bash_scripts/run_evaluation_array.sh
>   ```
> - To download the version of the file of a specific commit, replace `main` with `blob/<COMMIT-HASH>`:
>   ```
>   https://raw.githubusercontent.com/SainsburyWellcomeCentre/crabs-exploration/blob/<COMMIT-HASH>/bash_scripts/run_evaluation_array.sh
>   ```

4.  **Edit the evaluation bash script if required**

    Ideally, we won't make major edits to the bash scripts. If we find we do, then we may want to consider moving the relevant parameters to the config file, or making them a CLI argument.

    When launching an array job, we may want to edit the following variables in the bash script:

    - The `MLFLOW_CKPTS_FOLDER` and the `CKPT_FILENAME` variables, define which trained models we would like to evaluate. See the examples in the bash script comments for the syntax.
    - The number of trained models to evaluate needs to match the number of jobs in the array. To change the number of jobs in the array job, edit the line that start with `#SBATCH --array=0-n%m`. That command specifies to run `n` separate jobs, but not more than `m` at a time.
     - The `MLFLOW_FOLDER`. By default, we point to the "scratch" folder at `/ceph/zoo/users/sminano/ml-runs-all/ml-runs-scratch` . This folder holds runs that we don't need to keep. For runs we would like to keep, we will instead point to the folder at `/ceph/zoo/users/sminano/ml-runs-all/ml-runs`.

     Less frequently, one may need to edit:
    - The `EVALUATION_SPLIT`, to select whether we want to evaluate in the `test` set or the `validation` set.
    - the `GIT_BRANCH`, if we want to use a specific version of the 🦀 package. Usually we will run the version of the 🦀 package in `main`.

    Note that the dataset information and the config file used are retrieved from the MLflow logs of the corresponding training job. The experiment name for MLflow is created as the experiment name of the training job with the suffix `_evaluation`.

    If we would like a specific config file to be used (for example, to increase the number of workers in the evaluation), we can create a new config file and pass it to the evaluate command using the `--config_file` flag.

> [!CAUTION]
>
> If we launch a job and then modify the config file _before_ the job has been able to read it, we may be using an undesired version of the config in our job! To avoid this, it is best to wait until you can verify in MLflow that the job has the expected config parameters (and then edit the file to launch a new job if needed).


5. **Run the evaluation array job using the SLURM scheduler**

   To launch the evaluation array job, use the `sbatch` command and pass the path to the script:

   ```
   sbatch <path-to-run-evaluation-array-script>
   ```

6. **Check the status of the evaluation job**

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

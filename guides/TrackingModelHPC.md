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

    To do so, run any of the following commands. They will download a bash script for tracking (`run_tracking.sh` or `run_tracking_all_escape_events.sh`) to the current working directory.

    The download the version of these files in the `main` branch of the [🦀 repository](https://github.com/SainsburyWellcomeCentre/crabs-exploration), run one of the following commands.

    - To run video tracking on a specific video: download the `run_tracking.sh` file

      ```
      curl https://raw.githubusercontent.com/SainsburyWellcomeCentre/crabs-exploration/main/bash_scripts/run_tracking.sh > run_tracking.sh
      ```

    - To run video tracking on all escape events (or on a directory): download the `run_tracking_all_escape_events.sh` file

      ```
      curl https://raw.githubusercontent.com/SainsburyWellcomeCentre/crabs-exploration/main/bash_scripts/run_tracking_all_escape_events.sh > run_tracking_all_escape_events.sh
      ```

    These bash scripts will launch a SLURM job that:

    - gets the 🦀 package from git,
    - installs it in the compute node,
    - and runs a video tracking on a specific video.

> [!TIP]
> To retrieve a version of these files that is different from the files at the tip of `main`, edit the remote file path in the curl command:
>
> - For example, to download the version of the file at the tip of a branch called `<BRANCH-NAME>`, edit the path above to replace `main` with `<BRANCH-NAME>`:
>   ```
>   https://raw.githubusercontent.com/SainsburyWellcomeCentre/crabs-exploration/<BRANCH-NAME>/bash_scripts/run_tracking.sh
>   ```
> - To download the version of the file of a specific commit, replace `main` with `blob/<COMMIT-HASH>`:
>   ```
>   https://raw.githubusercontent.com/SainsburyWellcomeCentre/crabs-exploration/blob/<COMMIT-HASH>/bash_scripts/run_tracking.sh
>   ```

4.  **Edit the bash script!**

    For run the tracker, we need to ensure the correct trained model is used. All the parameters used in any training is logged into `mlflow`.

    We can see the perfomance of each training session by inspecting the `metrics` tab in `mlflow UI` where the `training loss`, `validation precision` and `validation recall` are plotted. The trained model (`checkpoint path`) are logged in `parameters` section under `overview` tab.

    When launching a tacking job, we may want to edit in the bash script:

    - The `TRAINED_MODEL_PATH`
    - The `OUTPUT_DIR`
    - The `VIDEO_PATH` (for `run_tracking.sh`) or `VIDEO_DIR` (for `run_tracking_all_escape_events.sh`)

    Less frequently, one may need to edit:

    - the `CONFIG_FILE`: usually we point to the same file we used to train the model at `/ceph/zoo/users/sminano/cluster_tracking_config.yaml` which we can edit.
    - the `GIT_BRANCH`, if we want to use a specific version of the 🦀 package. Usually we will run the version of the 🦀 package in `main`.

5.  **Other Inference options**

    By default, the inference will save the tracking output into a CSV file. There are other options that we can enable in CLI arguments:

    - `save_video` : This will save the tracking bounding boxes for every frame into a video output.
    - `save_frames` : This will save the corresponding frames to the CSV output. This is needed if we want to correct the tracking labels.

    Additionally, if we have ground truth for the video we used, we may want to add that to get the tracking evaluation:

    - `GT_PATH`

    We can add all these arguments in the bash script, for example:

    ```
    detect-and-track-video  \
    --checkpoint_path $CKPT_PATH \
    --video_path $VIDEO_PATH \
    --config_file $CONFIG_FILE \
    --gt_PATH $GT_PATH
    --device $DEVICE
    --save_video
    --save_frames
    ```

6.  **Run the inference job using the SLURM scheduler**

    To launch a job, use the `sbatch` command with the relevant training script:

    ```
    sbatch <path-to-inference-bash-script>
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

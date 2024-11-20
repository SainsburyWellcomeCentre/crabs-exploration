# Run detection and tracking over a set of videos in the cluster

1.  **Preparatory steps**

    - If you are not connected to the SWC network: connect to the SWC VPN.

2.  **Connect to the SWC HPC cluster**

    ```
    ssh <SWC-USERNAME>@ssh.swc.ucl.ac.uk
    ssh hpc-gw1
    ```

    It may ask for your password twice. To set up SSH keys for the SWC cluster, see [this guide](https://howto.neuroinformatics.dev/programming/SSH-SWC-cluster.html#ssh-keys).

3.  **Download the detect+track script from the 🦀 repository**

    To do so, run the following command, which will download a bash script called `run_detect_and_track_array.sh` to the current working directory.
    ```
    curl https://raw.githubusercontent.com/SainsburyWellcomeCentre/crabs-exploration/main/bash_scripts/run_detect_and_track_array.sh > run_detect_and_track_array.sh
    ```

    This bash script launches a SLURM array job that runs detection and tracking on an array of videos. The version of the bash script downloaded is the one at the tip of the `main` branch in the [🦀 repository](https://github.com/SainsburyWellcomeCentre/crabs-exploration).


> [!TIP]
> To retrieve a version of the file that is different from the file at the tip of `main`, edit the remote file path in the `curl` command:
>
> - For example, to download the version of the file at the tip of a branch called `<BRANCH-NAME>`, edit the path above to replace `main` with `<BRANCH-NAME>`:
>   ```
>   https://raw.githubusercontent.com/SainsburyWellcomeCentre/crabs-exploration/<BRANCH-NAME>/bash_scripts/run_detect_and_track_array.sh
>   ```
> - To download the version of the file of a specific commit, replace `main` with `blob/<COMMIT-HASH>`:
>   ```
>   https://raw.githubusercontent.com/SainsburyWellcomeCentre/crabs-exploration/blob/<COMMIT-HASH>/bash_scripts/run_detect_and_track_array.sh
>   ```

4.  **Edit the bash script if required**

    Ideally, we won't make major edits to the bash scripts. If we find we do, then we may want to consider moving the relevant parameters to the config file, or making them a CLI argument.

    When launching an array job, we may want to edit the following variables in the detect+track bash script:
    - The `CKPT_PATH` variable, which is the path to the trained detector model.
    - The `VIDEOS_DIR` variable, which defines the path to the videos directory.
    - The `VIDEO_FILENAME` variable, which allows us to define a wildcard expression to select a subset of videos in the directory. See the examples in the bash script comments for the syntax.
    - Remember that the number of videos to run inference on needs to match the number of jobs in the array. To change the number of jobs in the array job, edit the line that starts with `#SBATCH --array=0-n%m` and set `n` to the total number of jobs minus 1. The variable `m` refers to the number of jobs that can be run at a time.


    Less frequently, one may need to edit:
    - the `TRACKING_CONFIG_FILE`, which is the path to the tracking config to use. Usually we point to the file at `/ceph/zoo/users/sminano/cluster_tracking_config.yaml`, which we can edit.
    - the `OUTPUT_DIR_NAME`, the name of the output directory in which to save the results. By default it is created under the current working directory and named `tracking_output_slurm_<SLURM_ARRAY_JOB_ID>` (with `SLURM_ARRAY_JOB_ID` being the job ID of the array job).
    - the `SAVE_VIDEO` variable, which can be `true` or `false` depending on whether we want to to save the tracked videos or not. Usually set to `true`.
    - the `SAVE_FRAMES` variable, which can be `true` or `false` depending on whether we want to to save the untracked full set of frames per video or not. Usually set to `false`.
    - the `GIT_BRANCH`, if we want to use a specific version of the 🦀 package. Usually we will run the version of the 🦀 package in `main`.

    Currently, there is no option to pass a list of ground truth annotations that matches the set of videos analysed.

> [!CAUTION]
>
> If we launch a job and then modify the config file _before_ the job has been able to read it, we may be using an undesired version of the config in our job! To avoid this, it is best to wait until you can verify that the job has the expected config parameters (and then edit the file to launch a new job if needed).


5.  **Run the job using the SLURM scheduler**

    To launch a job, use the `sbatch` command with the relevant training script:

    ```
    sbatch <path-to-detect-and-track-bash-script>
    ```

6.  **Check the status of the job**

    To do this, we can:

    - Check the SLURM logs: these should be created automatically in the directory from which the `sbatch` command is run.
    - Run supporting SLURM commands (see [below](#some-useful-slurm-commands)).

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

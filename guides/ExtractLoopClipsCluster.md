# Run loop extraction over a set of videos in the cluster

1.  **Preparatory steps**

    - If you are not connected to the SWC network: connect to the SWC VPN.

2.  **Connect to the SWC HPC cluster**

    ```
    ssh <SWC-USERNAME>@ssh.swc.ucl.ac.uk
    ssh hpc-gw2
    ```

    It may ask for your password twice. To set up SSH keys for the SWC cluster, see [this guide](https://howto.neuroinformatics.dev/programming/SSH-SWC-cluster.html#ssh-keys).

3. **Fetch input csv file**

    We need to download the input csv file from the private [GIN repository](https://gin.g-node.org/SainsburyWellcomeCentre/CrabsField). To do that:
    1. Log into your GIN account running `gin login`
    2. If the `CrabsField` data repository exists locally, `cd` to it and run `gin download` to get the latest version
    3. If the `CrabsField` data repository does not exist locally, `cd` to the desired location and run `gin get SainsburyWellcomeCentre/CrabsField`.

    If there are issues with the files being downloaded as placeholder files, run `gin download --content`. See the [GIN guide](https://howto.neuroinformatics.dev/open_science/GIN-repositories.html#download-a-gin-dataset) for further details.

    Copy the full path to the spreadsheet at `CrabsField/crab-loops/loop-frames-ffmpeg.csv`, since we will need it to set the `CSV_PATH` variable in the bash script.


3.  **Download the extract-loops bash script from the 🦀 repository**

    To do so, run the following command, which will download a bash script called `run_extract_loop_clips_array.sh` to the current working directory.
    ```
    curl https://raw.githubusercontent.com/SainsburyWellcomeCentre/crabs-exploration/main/bash_scripts/run_extract_loop_clips_array.sh > run_extract_loop_clips_array.sh
    ```

    This bash script launches a SLURM array job that extracts loop clips on an array of videos. With the command above, he version of the bash script downloaded is the one at the tip of the `main` branch in the [🦀 repository](https://github.com/SainsburyWellcomeCentre/crabs-exploration).


> [!TIP]
> To retrieve a version of the file that is different from the file at the tip of `main`, edit the remote file path in the `curl` command:
>
> - For example, to download the version of the file at the tip of a branch called `<BRANCH-NAME>`, edit the path above to replace `main` with `<BRANCH-NAME>`:
>   ```
>   https://raw.githubusercontent.com/SainsburyWellcomeCentre/crabs-exploration/<BRANCH-NAME>/bash_scripts/run_extract_loop_clips_array.sh
>   ```
> - To download the version of the file of a specific commit, replace `main` with `blob/<COMMIT-HASH>`:
>   ```
>   https://raw.githubusercontent.com/SainsburyWellcomeCentre/crabs-exploration/blob/<COMMIT-HASH>/bash_scripts/run_extract_loop_clips_array.sh
>   ```


4.  **Edit the bash script if required**

    Review the bash script to ensure the following variables are set correctly:
    - `CSV_PATH`: path to the input csv file.
    - `INPUT_DIR`: path to the input directory containing the input videos.
    - `OUTPUT_DIR`: path to the output directory for the extracted loop clips.
    - `GIT_BRANCH`: version of the 🦀 package to use. Usually we will use the version at the tip of the `main` branch.
    - `VERIFY_FRAMES`: whether to verify frame count of the extracted clips matches the value in the csv file.


> [!CAUTION]
>
> If we launch a job and then modify the config file _before_ the job has been able to read it, we may be using an undesired version of the config in our job! To avoid this, it is best to wait until you can verify that the job has the expected config parameters (and then edit the file to launch a new job if needed).



5.  **Run the job using the SLURM scheduler**

    To launch a job, use the `sbatch` command with the path to the bash script:

    ```
    sbatch path/to/run_extract_loop_clips_array.sh
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

# Create zarr dataset from VIA track files in the cluster


1.  **Preparatory steps**

    - If you are not connected to the SWC network: connect to the SWC VPN.

2.  **Connect to the SWC HPC cluster**

    ```
    ssh <SWC-USERNAME>@ssh.swc.ucl.ac.uk
    ssh hpc-gw2
    ```

3. **Fetch metadata csv file**

    We need to download the metadata csv file from the private [GIN repository](https://gin.g-node.org/SainsburyWellcomeCentre/CrabsField). To do that:
    1. Log into your GIN account running `gin login`
    2. If the `CrabsField` data repository exists locally, `cd` to it and run `gin download` to get the latest version
    3. If the `CrabsField` data repository does not exist locally, `cd` to the desired location and run `gin get SainsburyWellcomeCentre/CrabsField`.

    If there are issues with the files being downloaded as placeholder files, run `gin download --content`. See the [GIN guide](https://howto.neuroinformatics.dev/open_science/GIN-repositories.html#download-a-gin-dataset) for further details.

    Copy the full path to the spreadsheet at `CrabsField/crab-loops/loop-frames-ffmpeg.csv`, since we will need it to set the `METADATA_CSV` variable in the bash script.

3.  **Download the create-zarr-dataset bash script from the 🦀 repository**

    To do so, run the following command, which will download a bash script called `run_create_zarr_dataset.sh` to the current working directory.
    ```
    curl https://raw.githubusercontent.com/SainsburyWellcomeCentre/crabs-exploration/main/bash_scripts/run_create_zarr_dataset.sh > run_create_zarr_dataset.sh
    ```

    This bash script launches a SLURM array job to create a zarr dataset from a set of input VIA track files. Each job in the array processes files from a single video. With the command above, the version of the bash script downloaded is the one at the tip of the `main` branch in the [🦀 repository](https://github.com/SainsburyWellcomeCentre/crabs-exploration).


> [!TIP]
> To retrieve a version of the file that is different from the file at the tip of `main`, edit the remote file path in the `curl` command:
>
> - For example, to download the version of the file at the tip of a branch called `<BRANCH-NAME>`, edit the path above to replace `main` with `<BRANCH-NAME>`:
>   ```
>   https://raw.githubusercontent.com/SainsburyWellcomeCentre/crabs-exploration/<BRANCH-NAME>/bash_scripts/run_create_zarr_dataset.sh
>   ```
> - To download the version of the file of a specific commit, replace `main` with `blob/<COMMIT-HASH>`:
>   ```
>   https://raw.githubusercontent.com/SainsburyWellcomeCentre/crabs-exploration/blob/<COMMIT-HASH>/bash_scripts/run_create_zarr_dataset.sh
>   ```

4. **Edit the bash script if required**

    Review the bash script to ensure the following variables are set correctly:
    - `VIA_TRACKS_DIR`: path to the input directory containing the VIA track files.
    - `METADATA_CSV`: path to the metadata csv file, derived in Step 3.

    Less frequently, you may also need to set the following variables:
    - `VIA_TRACKS_GLOB_PATTERN`: glob pattern to match the input VIA track files in the provided VIA tracks directory. By default, it is set to `"<VIDEO-NAME-WITHOUT-EXTENSION>.csv"`.
    - `ZARR_STORE_OUTPUT`: path to the output directory for the zarr dataset. By default, it is set to a filename that includes the SLURM array job ID. This ensures that different runs will generate different zarr datasets.
    - `ZARR_STORE_MODE`: determines whether to create a new zarr store (`'w'`) or to update an existing zarr store (`'a'`). By default it is set to `'a'`, since each job in the array processes files from a single video and we want to add the data of each video to the same zarr store.
    - `ZARR_MODE_GROUP`: determines whether to overwrite existing groups in the zarr store (`'w'`), update them (`'a'`), or throw an error if the group already exists (`'w-'`). By default, it is set to `'w-'`, since each job in the array should create a new group for its corresponding video and we would want to throw an error if a group already exists.
    - `GIT_BRANCH`: version of the 🦀 package to use. Usually we will use the version at the tip of the `main` branch.

5.  **Run the job using the SLURM scheduler**

    To launch a job, use the `sbatch` command with the path to the bash script:

    ```
    sbatch <path/to/run_create_zarr_dataset.sh>
    ```

6.  **Check the status of the job**

    To do this, we can:

    - Check the SLURM logs: these should be created automatically in the directory from which the `sbatch` command is run.
    - Run supporting SLURM commands (see [below](#some-useful-slurm-commands)).

> [!TIP]
> The `create-zarr-dataset` command first creates a temporary zarr store where each group (i.e. each subdirectory in the zarr store) is a video clip. It then restructures this store into a more convenient final version, in which each group is a video and all clips per video are concatenated. As a result, you may see temporary zarr datasets being created at the selected output location while the job is running (usually with a name such as `CrabTracks-slurm12345.zarr.task8.temp`, for a SLURM job with an array job ID `12345` and index `8`).

6. **Expected output**

    If the array job runs successfully, a zarr dataset named `CrabTracks-slurm<SLURM_ARRAY_JOB_ID>.zarr` will be generated in the location specified by `ZARR_STORE_OUTPUT`. Each group in the zarr dataset will correspond to a `movement` [bounding box dataset](https://movement.neuroinformatics.dev/latest/user_guide/movement_dataset.html) containing all tracks for one video.

    To inspect this dataset, please refer to the example notebooks in the 🦀 repository.

7. **Dealing with failed jobs**

    Sometimes some of the jobs in the array job fail due to non reproducible issues with `miniconda` in the cluster. This leads to an incomplete zarr store (e.g. called `store_1`), that contains only a subset of the videos we wanted to process. In those cases, this is the recommended approach to re-run the failed jobs and merge the results with a previous array job.

    1. If a job does not complete successfuly, its SLURM logs won't be copied across to the final zarr store. For traceability, it is recommended to first move the logs from the failed jobs to a separate directory under the `store_1` root directory (for example `store_1/logs_failed`). For example, for a failed job with array job ID `2382788` and job index `3`, we would run the following commands to copy the failed logs across:

    ```bash
    # create a directory for the failed logs
    mkdir /path/to/store_1.zarr/logs_failed

    # From the directory from which the `sbatch` command was run,
    # run the following command to move the failed logs across
    mv slurm_array.2382788-3.gpu-380-16.* /path/to/store_1.zarr/logs_failed/
    ```

    This will copy across both `.out` and `.err` logs for the failed job.

    2. Edit the bash script to run on the failed jobs only. First, edit the `#SBATCH --array=...` line in the bash script to specify the failed job indices only, as a comma-separate list (e.g., `#SBATCH --array=0,5,7-9%m` for failed jobs with indices 0, 5, 7, 8 and 9, with `m` being the maximum anumber of simultaneous jobs allowed). For more details about the syntax of the `--array` option, see the [SBATCH documentation](https://slurm.schedmd.com/sbatch.html#OPT_array).

    3. Next, comment out the if-clause in the `Check inputs` section of the bash script, which throws an error if the number of input VIA track files in the provided directory does not match the number of jobs in the array job. This is because we want to re-run only a subset of the jobs in the array, so the number of input files will be larger than the number of jobs and we need to skip this check.

    4. Run the edited bash script with `sbatch`. If the array job runs successfully, a new zarr store (that we will call `store_2` here) will be generated.

    5. Merge the two zarr stores. To do this, we first move the video directories from `store_2` to `store_1`. This can be done using the `mv` command or drag-and-dropping the folders in a file explorer.

> [!CAUTION]
> Remember to move across **just** the video directories (i.e. the zarr store groups), and not the metadata JSON file `zarr.json` at the root directory of `store_2`. Otherwise, we may overwrite the metadata JSON file of `store_1`!

    6. Finally, we update the metadata JSON file in the root group of `store_1` (i.e. the `zarr.json` file) to reflect the total number of videos processed. To do this, we can use the `zarr.consolidate_metadata` function, which updates the metadata JSON file based on the current structure of the zarr store:

    ```python
    import zarr
    zarr.consolidate_metadata("/path/to/store_1.zarr")
    ```

    7. We can use `xarray` to inspect the merged zarr store and verify that it contains the expected number of videos.

    ```python
    import xarray as xr
    dt = xr.open_datatree(path_store_1, engine="zarr", chunks={})
    print(f"Total groups: {len(dt)}") # should match the total number of videos processed
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

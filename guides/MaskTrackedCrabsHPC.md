# Mask tracked crabs in the cluster

This guide covers running `mask-tracked-crabs` over a whole trajectories zarr store in the SWC HPC cluster. The job prompts [SAM2](https://github.com/facebookresearch/sam2) with the tracked boxes already in the store, and writes a mask store holding one label image per frame. It needs no trained detector and no detection pass.

The job is a SLURM array with **one task per video**, so all videos are masked in parallel and each task appends its own group to a single shared output store.


1.  **Preparatory steps**

    - If you are not connected to the SWC network: connect to the SWC VPN.

2.  **Connect to the SWC HPC cluster**

    ```
    ssh <SWC-USERNAME>@ssh.swc.ucl.ac.uk
    ssh hpc-gw2
    ```

3.  **Check the inputs are in place**

    The job needs two things, both of which are outputs of earlier steps in the pipeline:

    - A **trajectories zarr store**, as written by `create-zarr-dataset` (see [CreateZarrDatasetForTracks.md](CreateZarrDatasetForTracks.md)). This is where the boxes that prompt SAM2 come from. Copy its full path, since we will need it to set the `BOXES_ZARR_STORE` variable in the bash script.

    - The **clip videos** the boxes refer to, as written by `extract-loop-clips` (see [ExtractLoopClipsCluster.md](ExtractLoopClipsCluster.md)), named `<video-id>-<clip-id>.mp4` — for example `04.09.2023-01-Right-Loop05.mp4`. These are the videos the store's clip-local `time` coordinate refers to, so they are the ones SAM2 reads pixels from. Copy the full path to the directory holding them, for the `CLIP_VIDEOS_DIR` variable.

    To check how many video groups the store holds, run:

    ```
    find <path-to-trajectories-store> -mindepth 1 -maxdepth 1 -type d -not -name "logs*" | wc -l
    ```

    This mirrors the equivalent check in the bash script. The bash script will launch an array job, and each job in the array will deal with one video.

4.  **Get the mask config file**

    The masking parameters live in their own config file. To get it locally (required), copy the default from the 🦀 repository to a location you can edit, for example `/ceph/zoo/users/sminano/cluster_mask_config.yaml`:

    ```
    curl https://raw.githubusercontent.com/SainsburyWellcomeCentre/crabs-exploration/main/crabs/mask/config/mask_config.yaml > /ceph/zoo/users/sminano/cluster_mask_config.yaml
    ```

    The parameters are described in the [masking README](../crabs/mask/README.md#parameters). Your file only needs the keys you want to change; the rest take the defaults. On the cluster, the two most likely to need tuning are:

    - `max_prompts_per_batch`: lower it if jobs fail with GPU out-of-memory errors.
    - `n_frames_per_shard`: trades memory per job against the number of files written to `/ceph`.

    > [!CAUTION]
    >
    > If we launch a job and then modify the config file _before_ the job has been able to read it, we may be using an undesired version of the config in our job! To avoid this, it is best to wait until you can verify that the job has the expected config parameters (and then edit the file to launch a new job if needed).

5.  **Download the bash script from the 🦀 repository**

    To do so, run the following command, which will download a bash script called `run_mask_array.sh` to the current working directory.
    ```
    curl https://raw.githubusercontent.com/SainsburyWellcomeCentre/crabs-exploration/main/bash_scripts/run_mask_array.sh > run_mask_array.sh
    ```

    This bash script launches a SLURM array job that masks the tracked crabs of a trajectories zarr store. Each job in the array masks every clip of a single video. With the command above, the version of the bash script downloaded is the one at the tip of the `main` branch in the [🦀 repository](https://github.com/SainsburyWellcomeCentre/crabs-exploration).

> [!TIP]
> To retrieve a version of the file that is different from the file at the tip of `main`, edit the remote file path in the `curl` command:
>
> - For example, to download the version of the file at the tip of a branch called `<BRANCH-NAME>`, edit the path above to replace `main` with `<BRANCH-NAME>`:
>   ```
>   https://raw.githubusercontent.com/SainsburyWellcomeCentre/crabs-exploration/<BRANCH-NAME>/bash_scripts/run_mask_array.sh
>   ```
> - To download the version of the file of a specific commit, replace `main` with `blob/<COMMIT-HASH>`:
>   ```
>   https://raw.githubusercontent.com/SainsburyWellcomeCentre/crabs-exploration/blob/<COMMIT-HASH>/bash_scripts/run_mask_array.sh
>   ```

6.  **Edit the bash script if required**

    Review the bash script to ensure the following variables are set correctly:
    - `BOXES_ZARR_STORE`: path to the input trajectories zarr store, from Step 3.
    - `CLIP_VIDEOS_DIR`: path to the directory with the clip videos, from Step 3.
    - `MASK_CONFIG_FILE`: path to the mask config file, from Step 4.

    Remember that the number of video groups in the input store needs to match the number of jobs in the array. To change the number of jobs, edit the line that starts with `#SBATCH --array=0-n%m` and set `n` to the total number of video groups minus 1. The variable `m` refers to the number of jobs that can run at a time.

    Less frequently, you may also need to set:
    - `MASK_ZARR_STORE_OUTPUT`: path to the output mask store. By default it is named after the SLURM array job ID, which ensures different runs generate different stores. **Every job in the array should point to the same store**, so this name must not depend on `SLURM_ARRAY_TASK_ID`.
    - `ZARR_MODE_STORE`: whether to create a new store (`'w'`) or add to an existing one (`'a'`). By default `'a'`, since each job in the array adds one video's masks to the same store.
    - `ZARR_MODE_GROUP`: whether to overwrite existing groups (`'w'`), update them (`'a'`), or throw an error if the group already exists (`'w-'`). By default `'w-'`, since each job should create a new group for its own video.
    - `GIT_BRANCH`: version of the 🦀 package to use. Usually we will use the version at the tip of the `main` branch.

    > [!NOTE]
    > Unlike the other bash scripts in the repository, this one installs the 🦀 package in two steps: first on its own, then with its `masks` extra, which adds SAM2 and `huggingface_hub`. SAM2's build imports torch, and in a single install uv would build SAM2 before torch is in the environment. Two details in the second step are deliberate:
    >
    > - **`--no-build-isolation-package sam-2`**, so SAM2 builds against the torch already in the environment. Without it, SAM2's build requirement on `torch>=2.5.1` pulls a whole second torch into an isolated build environment — possibly a different variant from the one we will actually run on. The same setting in the 🦀 `pyproject.toml` only applies in a checkout of the repository, not when installing the package from git.
    > - **`uv pip install setuptools` first.** Turning isolation off means SAM2's build requirements have to be satisfied in our own environment, and a `uv venv` starts with no setuptools. torch happens to depend on setuptools today, so this is belt-and-braces, but without it the failure is an opaque `No module named 'setuptools'` from inside the build backend.
    >
    > Each job in the array creates its own virtual environment, so that jobs do not race to create and install into a shared one. They do share the uv cache, though, and uv holds a lock on it while it fetches and builds SAM2 from git. With several jobs starting together on a slow shared filesystem, uv's default 300 s wait for that lock is not enough, so the script raises it with `UV_LOCK_TIMEOUT=3600`. Without it, the other jobs fail with `Timeout when waiting for lock`.
    >
    > The script also sets `HF_HOME` to a location on `/ceph/scratch`, so that the SAM2 checkpoint is downloaded once and shared across jobs rather than re-downloaded into each home directory.

    > [!NOTE]
    > The `#SBATCH --exclude` line keeps the jobs off the nodes with Quadro P5000 GPUs. These are too old for CUDA 13.0, which the installed torch is built against. If you change the partition or GPU request, keep this line or update it with any other nodes whose GPUs are not supported.

7.  **Run the job using the SLURM scheduler**

    To launch a job, use the `sbatch` command with the path to the bash script:

    ```
    sbatch path/to/run_mask_array.sh
    ```

8.  **Check the status of the job**

    To do this, we can:

    - Check the SLURM logs: these should be created automatically in the directory from which the `sbatch` command is run, and are moved into `<mask-store>/logs` when each job finishes.
    - Run supporting SLURM commands (see [below](#some-useful-slurm-commands)).

    Masking is the slowest step of the pipeline: expect **roughly 2 to 9 hours per video** at 10 fps, depending on the clip lengths, so the array job's wall time is set to 5 days per task. If your jobs are being cut short, raise the `#SBATCH -t` line.

9. **Expected output**

    If the array job runs successfully, a zarr store named `CrabMasks-slurm<SLURM_ARRAY_JOB_ID>.zarr` will be generated at the location specified by `MASK_ZARR_STORE_OUTPUT`, with one group per video:

    ```
    CrabMasks-slurm<SLURM_ARRAY_JOB_ID>.zarr/
    ├── <video_id>/                                  # one group per video
    │   ├── labels    (clip_id, time, img_h, img_w)  uint16
    │   ├── label_of  (individual,)                  uint16
    │   ├── clip_id     e.g. ["Loop00", "Loop05"]
    │   └── individual  e.g. ["id_0000", …]  — copied from the trajectories store
    ├── <video_id>_mask_config.yaml                  # the config each job ran with
    └── logs/
    ```

    `labels` holds **one label image per frame**: a single integer array where `0` is background and every other value identifies one crab. The `clip_id`, `time` and `individual` coordinates are the trajectories store's own, so the mask store and the trajectories store align 1:1 with no reindexing.

    To check the store holds every video you expected:

    ```python
    import xarray as xr
    dt = xr.open_datatree(path_to_mask_store, engine="zarr", chunks={})
    print(f"Total groups: {len(dt)}")  # should match the number of videos processed
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

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

        > [!WARNING]
        > The store must have been built **after** the `time` coordinate was made dense ([PR #291](https://github.com/SainsburyWellcomeCentre/crabs-exploration/pull/291)). In an older store, frames with no tracked crabs shifted every later frame earlier, so the masks would land on the wrong frames. `mask-tracked-crabs` checks this per clip and exits with a message naming the clip if the store predates it — rebuild the store with `create-zarr-dataset` in that case.

    - The **clip videos** the boxes refer to, as written by `extract-loop-clips` (see [ExtractLoopClipsCluster.md](ExtractLoopClipsCluster.md)), named `<video-id>-<clip-id>.mp4` — for example `04.09.2023-01-Right-Loop05.mp4`. These are the videos the store's clip-local `time` coordinate refers to, so they are the ones SAM2 reads pixels from. Copy the full path to the directory holding them, for the `CLIP_VIDEOS_DIR` variable.

    To check how many video groups the store holds, which is the number of tasks the array job will need:

    ```
    ls -d <path-to-trajectories-store>/*/ | wc -l
    ```

4.  **Stage the mask config file**

    The masking parameters live in their own config file, separate from the tracking config. Copy the default from the 🦀 repository to a location you can edit, for example `/ceph/zoo/users/sminano/cluster_mask_config.yaml`:

    ```
    curl https://raw.githubusercontent.com/SainsburyWellcomeCentre/crabs-exploration/main/crabs/tracker/config/mask_config.yaml > /ceph/zoo/users/sminano/cluster_mask_config.yaml
    ```

    The knobs are:

    - `sam2_model_id`: the Hugging Face model. By default `facebook/sam2.1-hiera-base-plus`. The `-tiny` and `-small` variants are considerably faster and may well be enough at this object size.
    - `max_prompts_per_batch`: how many box prompts go to SAM2 at once. It bounds peak GPU memory, because every prompt gets a full-frame float32 mask back — at 4K that is 35 MB per prompt, so the default of 32 peaks at about 1.1 GB. Lower it if you hit out-of-memory errors.
    - `shard_n_frames`: frames per shard file in the output store. It sets both the write buffer (`shard_n_frames × H × W × 2` bytes, so 566 MB at 32 frames and 4K) and the number of files. This is the one to tune per filesystem.
    - `occlusion_policy`: which crab keeps a pixel where two masks overlap. Currently `smallest_wins` is the only policy implemented.

    A config of your own need only name the keys it changes: the rest fall back to these defaults.

    > [!CAUTION]
    >
    > If we launch a job and then modify the config file _before_ the job has been able to read it, we may be using an undesired version of the config in our job! To avoid this, it is best to wait until you can verify that the job has the expected config parameters (and then edit the file to launch a new job if needed).

5.  **Download the masking bash script from the 🦀 repository**

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
    - Remember that the number of video groups in the input store needs to match the number of jobs in the array. To change the number of jobs, edit the line that starts with `#SBATCH --array=0-n%m` and set `n` to the total number of video groups minus 1. The variable `m` refers to the number of jobs that can run at a time.

    Less frequently, you may also need to set:
    - `MASK_ZARR_STORE_OUTPUT`: path to the output mask store. By default it is named after the SLURM array job ID, which ensures different runs generate different stores. **Every job in the array must point at the same store**, so this name must not depend on `SLURM_ARRAY_TASK_ID`.
    - `ZARR_MODE_STORE`: whether to create a new store (`'w'`) or add to an existing one (`'a'`). By default `'a'`, since each job in the array adds one video's masks to the same store.
    - `ZARR_MODE_GROUP`: whether to overwrite existing groups (`'w'`), update them (`'a'`), or throw an error if the group already exists (`'w-'`). By default `'w-'`, since each job should create a new group for its own video.
    - `GIT_BRANCH`: version of the 🦀 package to use. Usually we will use the version at the tip of the `main` branch.

    > [!NOTE]
    > Unlike the other bash scripts in the repository, this one installs a second package after the 🦀 package: SAM2 is an opt-in dependency, because it is not on PyPI. It is installed with `--no-build-isolation` so that it builds against the torch already in the environment, rather than pulling a whole second torch into an isolated build environment. The script also sets `HF_HOME` to a location on `/ceph/scratch`, so that the SAM2 checkpoint is downloaded once and shared across jobs rather than re-downloaded into each home directory.

7.  **Run the job using the SLURM scheduler**

    To launch a job, use the `sbatch` command with the path to the bash script:

    ```
    sbatch path/to/run_mask_array.sh
    ```

8.  **Check the status of the job**

    To do this, we can:

    - Check the SLURM logs: these should be created automatically in the directory from which the `sbatch` command is run, and are moved into `<mask-store>/logs` when each job finishes.
    - Run supporting SLURM commands (see [below](#some-useful-slurm-commands)).

    Masking is the slowest step of the pipeline: expect **roughly 2 to 9 hours per video** at 10 fps, depending on the clip lengths, so the array job's wall time is set to 1 day per task. If your jobs are being cut short, raise the `#SBATCH -t` line.

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

    A label image is the native input of both `skimage.measure.regionprops` and napari, so there is no conversion step on read:

    ```python
    import xarray as xr
    from skimage.measure import regionprops

    ds = xr.open_datatree(path_to_mask_store, engine="zarr", chunks={})["<video_id>"]

    v = int(ds.label_of.sel(individual="id_0003"))         # this crab's pixel value
    mask = ds.labels.sel(clip_id="Loop05") == v            # (time, img_h, img_w) bool
    regionprops(ds.labels.sel(clip_id="Loop05").isel(time=t).values)
    viewer.add_labels(ds.labels)                           # napari, sliders over clip and time
    ```

    > [!WARNING]
    > **Occlusion is resolved at write time, and the losses are not recorded.** A label image holds one crab per pixel, so where two masks overlapped the smaller crab kept the contested pixels and the larger one's are gone from the store. Measured on the trajectories store, 94% of crabs never overlap and 0.62% of box area is contested, but the tail is heavy: 2.4% of crabs lose more than a quarter of their box. Occluded crabs have to be found on read, by adjacency in `labels` or by comparing mask area against the tracked box area.

    For the rest of what the store cannot say for itself — in particular that `individual` names mean something only within one clip of one video, and how to decode `regionprops` output back to them — see [the masking section of the tracker README](../crabs/tracker/README.md#masking-tracked-crabs).

### Re-running failed jobs

Because each job in the array writes its own group, a failed job leaves the store simply missing that video, rather than corrupting it. To re-run the failed jobs into the **same** store:

1. **Edit the bash script to run the failed jobs only**

    First, edit the `#SBATCH --array=...` line to specify the failed task indices only, as a comma-separated list (e.g. `#SBATCH --array=0,5,7-9%m` for failed tasks 0, 5, 7, 8 and 9, with `m` the maximum number of jobs that can run simultaneously). For more details about the syntax of the `--array` option, see the [SBATCH documentation](https://slurm.schedmd.com/sbatch.html#OPT_array).

    Next, comment out the if-clause in the `Check inputs` section of the bash script, which throws an error if the number of video groups in the input store does not match the number of jobs in the array. We are re-running only a subset of the tasks, so this check needs to be skipped.

    > [!IMPORTANT]
    > Task indices are positions in the **sorted list of video groups in the input store**, which the script derives with `find ... | sort`. That list is the same on every run as long as the input store does not change, so the indices reported in the logs of the first run are still the right ones.

    Finally, set `MASK_ZARR_STORE_OUTPUT` to the **existing** store from the first run, rather than leaving it to be named after the new array job ID. With `ZARR_MODE_STORE="a"` and `ZARR_MODE_GROUP="w-"`, the re-run adds the missing groups and throws an error rather than silently overwriting one that already succeeded.

2. **Run the edited bash script with `sbatch`**

    ```bash
    sbatch path/to/edited/run_mask_array.sh
    ```

3. **Check the results**

    ```python
    import xarray as xr
    dt = xr.open_datatree(path_to_mask_store, engine="zarr", chunks={})
    print(f"Total groups: {len(dt)}")  # should match the total number of videos processed
    ```

    Note the logs of the two runs will both be under `<mask-store>/logs`, distinguished by their array job ID.

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

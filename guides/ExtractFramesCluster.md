# Run frame extraction in the SWC cluster

0. Connect to the SWC VPN

1. Connect to the SWC HPC

   ```
   ssh hpc-gw1
   ```

   It may ask for your password twice. To set up SSH keys for the SWC cluster, see [this guide](https://howto.neuroinformatics.dev/programming/SSH-SWC-cluster.html#ssh-keys)

2. Move to scratch and git clone this repo

   ```
   cd /ceph/scratch/sminano
   git clone https://github.com/SainsburyWellcomeCentre/crabs-exploration
   ```

   If the repo has already been git-cloned, `cd` to the repo directory and pull force to overwrite any local files:

   ```
   git fetch --all
   git reset --hard origin/main
   ```

   alternatively, to pull a specific branch:

   ```
   git reset --hard origin/<branch-name>
   ```

3. Edit the bash script if required

- The job array bash script should be at `/ceph/scratch/sminano/crabs-exploration/bash_scripts/run_frame_extraction_array.sh`
- Things to likely edit:
  - the list of input videos (one video per job in the array)
  - the name of the output directory that will hold the extracted frames (`OUTPUT_SUBDIR`)
  - ATTENTION!
    Check that the number of jobs in the array (`#SBATCH --array=0-1`) matches the number of videos to process
- Note: in the (hopefully near) future, we won't edit the bash script but rather an input config file whose content will be recorded in the logs.

4. Launch the array job with `sbatch` command

   From `/ceph/scratch/sminano/crabs-exploration/bash_scripts`, that would be

   ```
   sbatch run_frame_extraction_array.sh
   ```

5. Check the status of launched jobs

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

---

### Notes:

- When developing a new bash script, it is often a good idea to try each of the steps in an interactive node first!

  To request an interactive GPU node:

  ```
  srun -p gpu --gres=gpu:1 --pty bash -i
  ```

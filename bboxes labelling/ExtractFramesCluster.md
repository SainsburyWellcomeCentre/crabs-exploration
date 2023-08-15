# Run frame extraction in the SWC cluster

0. Connect to the SWC VPN

1. Connect to the SWC HPC

   ```
   ssh hpc-gw1
   ```

   It may ask for your password twice

2. Move to scratch and git clone this repo

   ```
   cd /ceph/scratch/sminano
   git clone https://github.com/sfmig/crabs-exploration.git
   ```

   If the repo has already been cloned, `cd` to the repo directory and pull force to overwrite any local files:

   ```
   git fetch --all
   git reset --hard origin/main
   ```

   alternatively, to pull a specific branch:

   ```
   git reset --hard origin/<branch-name>
   ```

3. Edit bash script if required

- The job array bash script should be at `/ceph/scratch/sminano/crabs-exploration/bboxes labelling\run_frame_extraction_array.sh`
- Things to likely edit:
  - the list of input videos (one video per job in the array)
  - the name of the output directory that will hold the extracted frames (`OUTPUT_SUBDIR`)
  - ATTENTION!
    Check that the number of jobs in the array (`#SBATCH --array=0-1`) matches the number of videos to process

4. Launch the array job with `sbatch` command from `/ceph/scratch/sminano/crabs-exploration/bboxes labelling`

   ```
   cd /Volumes/scratch/sminano/crabs-exploration/bboxes labelling
   sbatch run_frame_extraction_array.sh
   ```

5. Check status of running jobs

   ```
   squeue -u sminano
   ```

   To show details of the latest jobs (included completed or cancelled jobs)

   ```
   sacct -X -u sminano
   ```

   To check specific jobs by ID

   ```
   sacct -X -j 3813494,3813184
   ```

---

Notes:

- For a new bash script, it is often a good idea to try each of the steps with an interactive node first! To request a GPU one:
  ```
  srun -p gpu --gres=gpu:1 --pty bash -i
  ```
- To cancel a job
  ```
  scancel <jobID>
  ```
- To check timelimit
  ```
  squeue -u sminano --format="%i %P %j %u %T %l %C %S"
  ```

# Steps to run a SLEAP training job in the cluster

0. Connect to the SWC VPN (SonicWall)

1. Mount the SWC `ceph` filesystem on your laptop

   - Ideally, the .slp file with the annotated labels and the 'training package' will be in `ceph` already
     - to unzip the 'training package', run:
       ````
           unzip labels.v001.slp.training_job.zip -d labels.v001.slp.training_job
           ```
       ````
   - Remember to [change the permissions](https://askubuntu.com/questions/409025/permission-denied-when-running-sh-scripts) of the `.sh` files inside the zip
     ```
     chmod +x train-script.sh
     ```
     ```
     chmod +x inference-script.sh
     ```

2. Connect to the SWC cluster

   ```
   ssh <SWC-USERNAME>@ssh.swc.ucl.ac.uk
   ssh hpc-gw1
   ```

3. Navigate to the parent directory of the training job package, e.g.

   ```
   cd /ceph/zoo/users/sminano/crabs_pose_4k
   # this contains the labels.v001.slp.training_job directory
   ```

4. Create a batch job script to run the Inference job (e.g. `slurm_inference_TD.sh`)
   See for example:

   - [Niko's example](https://github.com/neuroinformatics-unit/swc-hpc-pose-estimation/blob/main/SLEAP/HowTo.md#model-inference)
   - [the sample script provided](https://github.com/sfmig/crabs-exploration/tree/main/pose%estimation/slurm_inference.sh),
   - or the example below.

   In `slurm_inference_TD.sh`, we would have something like:

   ```
   #!/bin/bash

   #SBATCH -p gpu # partition
   #SBATCH -N 1 # number of nodes
   #SBATCH --mem 12G # memory pool for all cores
   #SBATCH -n 2 # number of cores
   #SBATCH -t 3-04:00 # time (D-HH:MM)
   #SBATCH --gres gpu:1 # request 1 GPU (of any kind)
   #SBATCH -o slurm.%N.%j.out # write STDOUT
   #SBATCH -e slurm.%N.%j.err # write STDERR
   #SBATCH --mail-type=ALL
   #SBATCH --mail-user=s.minano@ucl.ac.uk

   # Load the SLEAP module
   module load SLEAP

   # training package directory
   DATA_DIR=/ceph/zoo/users/sminano/crabs_pose_4k_TD2
   JOB_DIR=$DATA_DIR/labels.v001.slp.training_job

   # inference video location
   INFER_DIR_NAME=Camera2
   INFER_VIDEO_NAME=NINJAV_S001_S001_T09.MOV
   INFER_VIDEO_PATH=/ceph/zoo/raw/CrabField/swc-courtyard_2023/$INFER_DIR_NAME/$INFER_VIDEO_NAME

   # Go to the training package directory
   cd $JOB_DIR

   # Run the inference command
   sleap-track $INFER_VIDEO_PATH \
   -m $JOB_DIR/models/230725_174219.centroid/training_config.json \
   -m $JOB_DIR/models/230725_174219.centered_instance/training_config.json \
   -o $INFER_DIR_NAME-$INFER_VIDEO_NAME.predictions.slp \
   --frames 4100-6600 \
   --verbosity json \
   --no-empty-frames \
   --tracking.tracker none \
   --gpu auto \
   --max_instances 1 \
   --batch_size 4
   ```

5. Launch job

   ```
   sbatch slurm_inference_TD.sh
   ```

6. Check status of the job

   ```
   squeue -u sminano

   ```

   Or

   ```
   sacct -X -u sminano

   ```

   Or to check job INFO details

   ```
   scontrol show jobid -dd <jobid>
   ```

---

To generate a rendering of a video with the bodypart predictions,
follow the same approach but running a rendering bash script instead - see [the sample script provided](https://github.com/sfmig/crabs-exploration/tree/main/pose%estimation/slurm_render.sh)

---

Some useful references:

- [How to](https://github.com/neuroinformatics-unit/swc-hpc-pose-estimation/blob/main/SLEAP/HowTo.md)by Niko
- Skeleton design clarifications [here](https://github.com/talmolab/sleap/issues/357#issuecomment-635134911)

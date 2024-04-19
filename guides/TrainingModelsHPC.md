# Train a detector in the cluster

0. If not in the SWC network: connect to the SWC VPN

1. Connect to the SWC HPC

   ```
   ssh hpc-gw1
   ```

   It may ask for your password twice. To set up SSH keys for the SWC cluster, see [this guide](https://howto.neuroinformatics.dev/programming/SSH-SWC-cluster.html#ssh-keys)

2. Clone this repository somewhere in `ceph` (e.g., in your scratch partition):

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

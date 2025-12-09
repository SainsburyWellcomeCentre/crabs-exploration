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
    We need to download the input csv file from the private GIN repository. To do that:
    1. Log into your GIN account with `gin login`
    2. If the `CrabsField` data repository exists locally, `cd` to it and run `gin download` to get the latest version
    3. If the `CrabsField` data repository does not exist locally, `cd` to the desired location and run `gin get SainsburyWellcomeCentre/CrabsField`.
    
    If there are issues with the files being downloaded as placeholder files, run `gin download --content`. See the [GIN guide](https://howto.neuroinformatics.dev/open_science/GIN-repositories.html#download-a-gin-dataset) for further details.
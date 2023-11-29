# Create a GIN private repository

GIN is a free and open data management system designed for reproducible management of neuroscientific data. It is a web-accessible repository store of your data based on `git` and `git-annex` that you can access securely anywhere you desire while keeping your data in sync, backed up and easily accessible.

Below we explain the steps to create a GIN private repository for a dataset (in our case, a set of labelled images). Note that all GIN repos are private by default.

## Preparatory steps - do only once

These steps apply to any of the workflows below, but we need to them only the first time we use GIN in our machine.

1. Create a GIN account
2. [Download GIN CLI](https://gin.g-node.org/G-Node/Info/wiki/GIN+CLI+Setup#setup-gin-client) and set it up by running:
   ```
   gin login
   ```
3. Confirm that everything is working properly by typing:
   ```
   gin --version
   ```

## Create a GIN private repository from the command line (CLI)

1. **Log in to the GIN server**

   Before running any `gin` commands, make sure you are logged in to your account by running:

   ```
   gin login
   ```

   - You will be prompted for your GIN username and password.
   - To list the repositories available to your account: `gin repos --all`

> [!TIP]
>
> You may need `sudo` permissions for some of the following `gin` commands. If so, remember to prepend all commands with `sudo`.

2. **Initialise a GIN repository**

   - **Option 1: on a new directory**

     - Create a new GIN repository locally and on the GIN server by running:
       ```
       gin create <repository name>
       ```
         <details><summary> OR alternatively:</summary>

         Create a repository in the GIN server [from the browser](https://gin.g-node.org/repo/create), and download it locally to your local workspace by running:
         ```
         gin get <user name>/<repository name>
         ```
         </details>
     - Once the repository has been initialised, add data to the new local GIN repository with `mv`, `cp` or drag-and-dropping files to the directory.

   - **Option 2: on an existing directory**

     - To create a new repository on the GIN server and in the current working directory in one go, run:
       ```
       gin create --here <name>
       ```
        <details><summary> OR alternatively:</summary>

        To do each step independently:
        - Initialise the current working directory as a GIN repository by running:
        ```
        gin init
        ```
        - Then add a remote for your GIN local repository by running:
        ```
        gin add-remote <name> <location>
        ```
        where `<name>` is the name you want to give to the remote (e.g. `origin`) and `<location>` is the location of the data store, which should be in the form of alias:path or server:path (e.g. `gin add-remote origin gin:sfmig/crab-data`). - If the remote GIN repository doesn't exist, you will be prompted to either create the remote GIN repository, add the remote address anyways or abort. - To show the remotes accessible to your GIN account run `gin remotes`.
        </details>

   > [!TIP]
   >
   > To create a GIN repository on a `ceph` directory:
   >
   > - You may need to mount the `ceph` directory first. To do this temporarily (i.e., until the next reboot), follow [this guide](https://howto.neuroinformatics.dev/programming/Mount-ceph-ubuntu-temp.html). To do this permanently, follow [this one](https://howto.neuroinformatics.dev/programming/Mount-ceph-ubuntu.html).
   > - You may also need to add an exception for the mounted directory. To do so, run the following command:
   >
   >   ```
   >   git config --global --add safe.directory /mnt/<path-to-the-mounted-directory>
   >   ```
   >
   > - Alternatively, you can log to SWC's HPC cluster (specifically, its [gateway node](https://howto.neuroinformatics.dev/_images/swc_hpc_access_flowchart.png) `hpc-gw1`), which has the GIN CLI client installed, and work from there. This is likely faster than mounting the `ceph` directory in your laptop, since the cluster is in the same network as `ceph` (and actually physically close to it).

3. **Add files to the GIN remote repository**

   It is good practice to keep a record of the changes in the repository through commit messages. To keep a useful and clean commit history, it is also recommended to make small commits by selecting a subset of the files.

   - To make a record of the current state of a local repository, run

      ```
      gin commit --message <message> <filename>
      ```

      You can replace the `filename` above by an expression with wildcard (e.g., `*.png` to include all png files). It can also be a list of files (separated by white spaces). A filename equal to `.` will include all files with changes. See the full syntax [here](https://gin.g-node.org/G-Node/Info/wiki/GIN+CLI+Help#record-changes-in-local-repository).

   - To upload all local changes to the remote GIN repository:

      ```
      gin upload <filename>
      ```

      Like before, `filename` accepts wildcards, can be a list of files (separated by white spaces) and can be replaced by `.` to include all files with changes. Again, the recommended practice would be to upload data in small-ish chunks. You can run an upload command after a few commits (so not necessarily after each commit).

      If the set of files you upload includes files that have been changed locally but not committed, they will be automatically committed when uploading. See full syntax [here](https://gin.g-node.org/G-Node/Info/wiki/GIN+CLI+Help#upload-local-changes-to-a-remote-repository).

      Note this command sends all changes made in the directory to the server, including deletions, renames, etc. Therefore, if you delete files from the directory on your computer and perform a gin upload, the deletion will also be sent and the file will be removed from the server as well. Such changes can be synchronized without uploading any new files by not specifying any files or directories (i.e. simply running `git upload`). See further details in [the docs basic workflow](https://gin.g-node.org/G-Node/Info/wiki/GIN+CLI+Usage+Tutorial#basic-workflow-only-using-gin).

> [!TIP]
>
> - Use `gin ls` to check on the current status of the GIN repository - this is somewhat equivalent to `git status`
> - Use `gin sync` to sync the changes bi-directionally between the local and the remote GIN repository.
> - If the output from `gin ls` doesn't look right (e.g., files already uploaded to the GIN server appear under `Locally modified (unsaved)`), try running `gin sync` and check the status again.

### To update a dataset that is hosted in GIN

1. To clone (retrieve) a repository from the remote server to a local machine:

   ```
   gin get <relative-path>
   ```

> [!TIP]
> To see the relative paths accesible from your GIN account, run `gin repos`

2. Add files to the directory where the local repository is in, and commit them:

   ```
   gin commit -m <message> <filename>
   ```

3. Upload the committed changes to the GIN server with:
   ```
   gin upload <filename>
   ```

### To download the data locally

- To download changes from the remote repository to the local clone, and get the most updated repository, run:

  ```
  gin download
  ```

  This command is somewhat equivalent to "pulling" the latest changes to the repository. It will create new files that were added remotely, delete files that were removed, and update files that were changed. With the `--content` flag, it optionally downloads the content of all files in the repository. If 'content' is not specified, new files will be empty placeholders. Content of individual files can later be retrieved using the 'get content' command, and later removed with 'remove content'. See [the docs](https://gin.g-node.org/G-Node/Info/wiki/GIN+CLI+Help#download-all-new-information-from-a-remote-repository) for further details.

- To retrieve the content of an individual file, run:

  ```
  gin get-content <filename>
  ```

- To donwload the data programmatically in your Python code

  We recommend using [pooch](https://www.fatiando.org/pooch/latest/index.html) to easily download data from the GIN repo's URL. Pooch also has some nice bonus functionalities like caching the downloaded data, verifying cryptographic hashes or unzipping files upon download.

### Other useful tips

- To [unannex a file](https://gin.g-node.org/G-Node/Info/wiki/FAQ+Troubleshooting#how-to-unannex-files), aka remove a file from the GIN tracking before uploading:

  ```
  gin git annex unannex [path/filename]
  ```

- To stop tracking the GIN repo locally delete the `.git` directory

- To delete a GIN repository but keep the git repo:

  - delete the repository in the GIN server via the browser
  - delete the GIN local repository with `git annex uninit`
    - this command removes relevant bits in `.git/annex` and `.git/objects`, but some pre-commits may need to be edited by hand (see this [SO post](https://stackoverflow.com/questions/24447047/remove-git-annex-repository-from-file-tree)).

- To lock / unlock the data

### Helpful resources

- [GIN CLI Usage tutorial](https://gin.g-node.org/G-Node/Info/wiki/GIN+CLI+Usage+Tutorial): includes a description of a basic workflow and examples of multi-user workflows.
- [GIN commands cheatsheet](https://gin.g-node.org/G-Node/Info/wiki/GIN+CLI+Help)
- [Troubleshooting](https://gin.g-node.org/G-Node/Info/wiki/FAQ%20Troubleshooting)
- [GIN CLI Recipes](https://gin.g-node.org/G-Node/Info/wiki/GIN+CLI+Recipes)

### References

- https://movement.neuroinformatics.dev/community/contributing.html#adding-new-data
- https://gin.g-node.org/G-Node/info/wiki#how-do-i-start
- https://gin-howto.readthedocs.io/en/latest/gin-repositories.html
- On GIN and its relation to `git-annex` (very high-level): https://gin.g-node.org/G-Node/Info/wiki/GIN+Advantages+Structure

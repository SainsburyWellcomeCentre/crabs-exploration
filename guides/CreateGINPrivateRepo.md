# Create a GIN private repository

[GIN](https://gin.g-node.org/G-Node/Info/wiki) (hosted by the German Neuroinformatics Node) is a free and open data management system designed for reproducible management of neuroscientific data. It is a web-accessible repository store of your data based on `git` and `git-annex` that you can access securely anywhere you desire while keeping your data in sync, backed up and easily accessible.

Below we explain the steps to create a GIN private repository for a dataset (in our case, a set of labelled images).

> [!NOTE]
> All GIN repos are private by default.

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
       gin create --here <repository name>
       ```

        <details><summary>Or, to do each step independently:</summary>

       - Initialise the current working directory as a GIN repository by running:

       ```
       gin init
       ```

       - Then add a remote for your GIN local repository by running:

       ```
       gin add-remote <name> <location>
       ```

       where `<name>` is the name you want to give to the remote (e.g. `origin`) and `<location>` is the location of the data store, which should be in the form of alias:path or server:path (e.g. `gin add-remote origin gin:sfmig/crab-data`).

       - If the remote GIN repository doesn't exist, you will be prompted to either create the remote GIN repository, add the remote address anyways or abort.
       - To show the remotes accessible to your GIN account run `gin remotes`.
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

     After running `gin upload`, the data will be uploaded to the GIN server and it will be possible to fetch it later from there. However, note this command sends all changes made in the directory to the server, including deletions, renames, etc. Therefore, if you delete files from the directory on your computer and perform a gin upload, the deletion will also be sent and the file will be removed from the server as well. Such changes can be synchronized without uploading any new files by not specifying any files or directories (i.e. simply running `git upload`). See further details in [the docs basic workflow](https://gin.g-node.org/G-Node/Info/wiki/GIN+CLI+Usage+Tutorial#basic-workflow-only-using-gin).

> [!TIP]
>
> - Use `gin ls` to check on the current status of the GIN repository - this is somewhat equivalent to `git status`. The acronyms for the different status of the files are described [here](https://gin.g-node.org/G-Node/Info/wiki/GIN+CLI+Help#list-the-sync-status-of-files-in-the-local-repository)
> - Use `gin sync` to sync the changes bi-directionally between the local and the remote GIN repository.
> - If the output from `gin ls` doesn't look right (e.g., files already uploaded to the GIN server appear under `Locally modified (unsaved)`), try running `gin sync` and check the status again.

## To update a dataset that is hosted in GIN

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

## To download the dataset locally

If the repository doesn't exist locally:

1. Clone (retrieve) the repository from the remote server to your local machine:

   ```
   gin get <relative-path>
   ```

   The large files will be downloaded as placeholders.

2. To download the content of the placeholder files locally, run:
   ```
   gin download --content
   ```
   If the large files in the dataset are locked, this command will turn the placeholder files into symlinks which point to the actual content stored in the git annex subdirectory. If the files are not locked, this command will replace the placeholder files by the full-content files and also download the git annex content locally. See the section on [File locking](#file-locking) for further details.

If the repository already exists locally:

1. Download any changes from the remote repository to the local clone, and get the most updated repository, by running (from the GIN repository directory):

   ```
   gin download
   ```

   In this context this command is somewhat equivalent to "pulling" the latest changes to the repository. It will create new files that were added remotely, delete files that were removed, and update files that were changed.

   With the `--content` flag, it optionally downloads the content of all files in the repository. If `--content` is not specified, new files will be empty placeholders.

   Content of individual files can be retrieved using the `gin get-content <filename>` command, and later removed with `gin remove-content <filename>`. See [the docs](https://gin.g-node.org/G-Node/Info/wiki/GIN+CLI+Help#download-all-new-information-from-a-remote-repository) for further details.

To download the data programmatically in your Python code:

- We recommend using [pooch](https://www.fatiando.org/pooch/latest/index.html) to easily download data from the GIN repo's URL. Pooch also has some nice bonus functionalities like caching the downloaded data, verifying cryptographic hashes or unzipping files upon download.

## Other useful tips

- To [unannex a file](https://gin.g-node.org/G-Node/Info/wiki/FAQ+Troubleshooting#how-to-unannex-files), aka remove a file from the GIN tracking before uploading:

  ```
  gin git annex unannex [path/filename]
  ```

- To stop tracking the GIN repo locally delete the `.git` directory

  > [!NOTE]
  > If in the GIN repo the files are locked, remember to unlock them before removing the `.git` directory! Otherwise we won't be able to delete the `.git/annex` content.

- To delete a GIN repository but keep the git repo:

  - delete the repository in the GIN server via the browser
  - delete the GIN local repository with `git annex uninit`
    - this command removes relevant bits in `.git/annex` and `.git/objects`, but some pre-commits may need to be edited by hand (see this [SO post](https://stackoverflow.com/questions/24447047/remove-git-annex-repository-from-file-tree)).

## File locking

[File locking](https://gin.g-node.org/G-Node/Info/wiki/GIN+CLI+Usage+Tutorial#file-locking) is an important point in GIN repos and git-annex which surprisingly comes up quite late in the docs. Below, are the main ideas behind this.

- Files in a GIN repo can be locked or unlocked.

- The lock state of a file is persistent. This means that if I clone a GIN repo whose files are unlocked, I lock them in my local copy, and then upload that to the GIN server, the next time someone clones from the GIN repo the files they fetch will be locked.

- The lock state actually relates more to the nature of the placeholder file (the file in the working directory when we do `gin get <repository`):

  - on Unix-like systems: if a file is locked, its corresponding placeholder file will be a symlink pointing to the annexed content (i.e., the content under `.git/annex/objects`). This way we can open and inspect the file but not modify it. If a file is unlocked, the placeholder file in the working directory is an ASCII text file with a path. This path is _approximately_ where the content of the file will be downloaded to when we request it.
  - on Windows: from the docs, if a file is locked, the placeholder file is a plain text file pointing to the content in the git annex. If a file is unlocked, presumably the behaviour is the same as in Unix-like systems. I haven't tested the situation on Windows.

- Unlocked files can be edited. If the data is unlocked and the full content of the dataset is downloaded locally, the file in the working directory has content, and so does its copy under git annex.

> [!CAUTION]
> This doubles disk usage of files checked into the repo, but in exchange users can modify and revert files to previous commits.

- Locked files cannot be edited. For example, if we open a locked image with Preview in MacOS and try to edit it, we will be asked if we wish to unlock the file. However even if we do, we won't be able to save any changes because we don't have writing permissions. Files need to be committed before locking.

- We can switch the state for one or more file using `gin lock <filename>` and `gin unlock <filename>`. After changing the state, remember to record the new state with `gin commit`!

- Recommendations from the docs on when to lock / unlocked data:
  - Keep files _unlocked_ if the workflow requires editing large files and keeping snapshots of the progress. But keep in mind this will increase storage use with every commit of a file.
  - Keep files _locked_ if using the repo mainly as long term storage, as an archive, if files are only to be read and if the filesystem supports symlinks. This will save extra storage of keeping two copies of the same file.

## Some under-the-hood details...

- GIN is a wrapper around [git-annex](https://git-annex.branchable.com/)

- The high-level idea behind git-annex is:
  - git is designed to track small text files, and doesn't cope well with large binary files
  - git-annex bypasses this by using git only to track the names and metadata of these large binary files, but not their content
  - the content of these files is only retrieved on demand
- How? Case for an unlocked dataset

  - When we `gin download` a repository from the GIN server, we get a local "copy" (clone) of the dataset in our machine. It is not strictly a copy, because the large binary files that make up this dataset will only be downloaded as placeholders.

  - These placeholder files have the same filenames (and paths) as the corresponding original files, but are instead simply ASCII text files (if the data is unlocked). If we open these placeholder files, we see they contain a path. This path is where the actual content of the corresponding file will be downloaded to, when we request it.
  - For example, if the placeholder ASCII text file with name `09.08_09.08.2023-01-Left_frame_013230.png` points to this path `/annex/objects/MD5-s15081575--f0a21c00672ab7ed0733951a652d4b49`, it means that when we specifically request for this file's content with `gin get-content 09.08_09.08.2023-01-Left_frame_013230.png`, the actual png file will be downloaded to `.git/annex/objects/Xq/7G/MD5-s15081575--f0a21c00672ab7ed0733951a652d4b49/MD5-s15081575--f0a21c00672ab7ed0733951a652d4b49` (note that the path in the ASCII file and the actual path are somewhat different, since the actual path contains some subdirectories under `objects`). We can actually verify this file is the image by opening it with an image viewer (in mac: `open -a Preview .git/annex/objects/Xq/7G/MD5-s15081575--f0a21c00672ab7ed0733951a652d4b49/MD5-s15081575--f0a21c00672ab7ed0733951a652d4b49`)

- How? Case for a locked dataset

  - When we `gin download` a repository from the GIN server, we get a local "copy" (clone) of the dataset in our machine. It is not strictly a copy, because the large binary files that make up this dataset will only be downloaded as placeholders.
  - If the data is locked and no content has been downloaded, the symlinks in the working directory will be broken (since there is no data in the git annex to retrieve).
  - To get the actual content in the git annex, we need to run `gin download --content`. This will fetch the content from the GIN server (presumably). After this, the symlinks in the working directory should work

- And in an existing directory?

  - After initialising the GIN repo in the current directory and adding a remote, we would commit the data. When committing, the data is "copied" to the git annex.
  - To replace the copies in the working directory with symlinks to the git annex content, we run `gin lock <path-to-data>`
  - If we commit this state change and upload the changes to the GIN server, the files will be locked for any future clones of the repo.

- Useful tools for inspecting this further

  - `file` shows the type of file (inspecting the file, rather than plainly looking at the extension like Finder does)
  - `open -a Preview <>` to open a png file that has no extension
  - `ls -l <path to symlink` to check the path a symlink points to

## Helpful resources

- [GIN CLI Usage tutorial](https://gin.g-node.org/G-Node/Info/wiki/GIN+CLI+Usage+Tutorial): includes a description of a basic workflow and examples of multi-user workflows.
- [GIN commands cheatsheet](https://gin.g-node.org/G-Node/Info/wiki/GIN+CLI+Help)
- [Troubleshooting](https://gin.g-node.org/G-Node/Info/wiki/FAQ%20Troubleshooting)
- [GIN CLI Recipes](https://gin.g-node.org/G-Node/Info/wiki/GIN+CLI+Recipes)

## References

- https://movement.neuroinformatics.dev/community/contributing.html#adding-new-data
- https://gin.g-node.org/G-Node/info/wiki#how-do-i-start
- https://gin-howto.readthedocs.io/en/latest/gin-repositories.html
- On GIN and its relation to `git-annex` (very high-level): https://gin.g-node.org/G-Node/Info/wiki/GIN+Advantages+Structure

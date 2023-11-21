# Create a GIN private repository for the labelled data

Below we explain the steps to follow to create a GIN private repository for a set of labelled data

### Preparatory steps - do only once

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

### To set up a GIN private repository from the CLI

All GIN repos are private by default.
Before running any `gin` commands, make sure you are logged in to your account (via `gin login`).

1. Use `gin login` to confirm your identity with the GIN server

   - You will be prompted for your GIN username and password.
   - To list the repositories available to your account: `gin repos --all`

> [! TIP]
>
> You may need `sudo` permissions for some of the following `gin` commands. If so, remember to prepend all commands with `sudo`

2. Create a GIN repository

   - Option 1: on a new directory

     - Run `gin create <repository name>` to create a new GIN repository locally and on the GIN server, or
     - Create a repository in the GIN server [from the browser](https://gin.g-node.org/repo/create), and run `gin get <user name>/<repository name>` locally to download it to your local workspace.
     - Add data (with `mv`, `cp` or drag-and-drop) to the new local GIN repository

   - Option 2: on an existing directory
     - from the relevant directory, run `gin init` to initialise the current working directory as a GIN repository
     - Optional: create a repository in the GIN server [from the browser](https://gin.g-node.org/repo/create) ----> this doesnt work well for me!
     - add a remote with `gin add-remote <name> <location>`, with the location of the data store in the form of alias:path or server:path
       - e.g: `gin add-remote origin gin:sfmig/crab-data`
       - select create (if existing it doesnt detect it?)
       - To show remotes: `gin remotes`.

> [! TIP]
>
> To create a GIN repository on a `ceph` directory, you may need to mount the `ceph` directory first. To do this temporarily (i.e., until the next reboot), follow [this guide](https://howto.neuroinformatics.dev/programming/Mount-ceph-ubuntu-temp.html). To do this permanently, follow [this one](https://howto.neuroinformatics.dev/programming/Mount-ceph-ubuntu.html).
>
> You may also need to add an exception for the mounted directory. To do so, run the following command:
>
> ```
> git config --global --add safe.directory /mnt/<path-to-the-mounted-directory>
> ```

3. Record changes and upload data to the GIN remote repository

   - To make a record of the current state of a local repository, run
     ```
     gin commit --message <message> <filename>
     ```
     You can replace the `filename` above by `.` to include all files with changes. See the full syntax [here](https://gin.g-node.org/G-Node/Info/wiki/GIN+CLI+Help#record-changes-in-local-repository)
   - To upload all local changes to the remote GIN repository:
     ```
     gin upload <filename>
     ```
     Like before, you can replace the `filename` above by `.` to include all files with changes. See full syntax [here](https://gin.g-node.org/G-Node/Info/wiki/GIN+CLI+Help#upload-local-changes-to-a-remote-repository)

> [! TIP]
>
> Use `gin ls` to check on the current status of the GIN repository, somewhat equivalent to `git status`

### To update a dataset that is hosted in GIN

1. Clone the data repository
   ```
   gin get <relative-path>
   ```
2. Add files and commit them
   ```
   gin commit -m <message> <filename>
   ```
3. Upload the committed changes to the GIN server
   ```
   gin upload
   ```

- To get the most updated repository, run: `gin download`

> [! NOTE]
> From [the docs](https://gin.g-node.org/G-Node/Info/wiki/GIN+CLI+Help#download-all-new-information-from-a-remote-repository) :
> `gin download` downloads changes from the remote repository to the local clone. This will create new files that were added remotely, delete files that were removed, and update files that were changed.
>
> With the `--content` flag, it optionally downloads the content of all files in the repository. If 'content' is not specified, new files will be empty placeholders. Content of individual files can later be retrieved using the 'get content' command, and later removed with 'remove content'.

- To sync the changes bi-directionally, run: `gin sync`

> [! TIP]
>
> If the output from `gin ls` doesn't look right (e.g., files already uploaded to the GIN server appear under `Locally modified (unsaved)`), try running `gin sync` and check the status again.

### Other useful commands

- To undo a commit
  ```
  ...
  ```
- To ammend a commit message
  ```
  ...
  ```
- To stop tracking the GIN repo locally: delete the `.git` directory

- To delete a GIN repository but keeping git repo:
  - delete the repository in the GIN server through the browser and then log back into gin?
  - delete the GIN local repository with `git annex uninit`
    - this command removes relevant bits in `.git/annex` and `.git/objects`, but some pre-commits may need to be edited by hand
    - see this [SO post](https://stackoverflow.com/questions/24447047/remove-git-annex-repository-from-file-tree)

To fetch the data in your code
...

To extract the checksum

> Determine the sha256 checksum hash of each new file, by running sha256sum <filename> in a terminal. Alternatively, you can use pooch to do this for you: python -c "import pooch; pooch.file_hash('/path/to/file')". If you wish to generate a text file containing the hashes of all the files in a given folder, you can use python -c "import pooch; pooch.make_registry('/path/to/folder', 'sha256_registry.txt').

### Useful resources

- [GIN CLI Usage tutorial](https://gin.g-node.org/G-Node/Info/wiki/GIN+CLI+Usage+Tutorial)---> best ref probably
- [GIN commands cheatsheet](https://gin.g-node.org/G-Node/Info/wiki/GIN+CLI+Help)
- [Troubleshooting](https://gin.g-node.org/G-Node/Info/wiki/FAQ%20Troubleshooting)
- [GIN CLI Recipes](https://gin.g-node.org/G-Node/Info/wiki/GIN+CLI+Recipes)

### References

- https://movement.neuroinformatics.dev/community/contributing.html#adding-new-data
- https://gin.g-node.org/G-Node/info/wiki#how-do-i-start
- https://gin-howto.readthedocs.io/en/latest/gin-repositories.html

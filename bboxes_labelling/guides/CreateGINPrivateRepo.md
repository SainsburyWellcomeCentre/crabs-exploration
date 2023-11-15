# Create a GIN private repository for the labelled data

Refs:

- https://movement.neuroinformatics.dev/community/contributing.html#adding-new-data
- https://gin.g-node.org/G-Node/info/wiki#how-do-i-start
- https://gin-howto.readthedocs.io/en/latest/gin-repositories.html

[GIN CLI Usage tutorial](https://gin.g-node.org/G-Node/Info/wiki/GIN+CLI+Usage+Tutorial)---> best ref probably
[Cheatsheet](https://gin.g-node.org/G-Node/Info/wiki/GIN+CLI+Help)
[Troubleshooting](https://gin.g-node.org/G-Node/Info/wiki/FAQ%20Troubleshooting)

### Prep - only once

1. Create a GIN account
2. [Download GIN CLI](https://gin.g-node.org/G-Node/Info/wiki/GIN+CLI+Setup#setup-gin-client) and set it up with `gin login`
3. Confirm that everything is working properly by typing the command `gin --version`.

### To set up a GIN private repository from the CLI

All GIN repos are private by default

1. Use `gin login` to confirm your identity with the GIN server (This requires the username and password from your GIN server account).

   - To list available repositories: `gin repos --all`

2. Create a GIN repository
   - Option 1: on a new directory
     - Use `gin create <repository name>` to create a new repository on the GIN server AND locally, or
     - Create a repository in the GIN server [from the browser](https://gin.g-node.org/repo/create, and run `gin get <user name>/<repository name>` locally to download it to your local workspace.
     - Add data (`mv` or `cp`) to the new directory
   - Option 2: on an existing directory
     - from the relevant directory, run `gin init` to initialise the cwd as a GIN repository
     - add a remote with `gin add-remote <name> <location>`, e.g: `gin add-remote origin gin:sfmig/crab-data`
       - To show remotes: `gin remotes`.
3. Add data to the repository
   - To record changes made in a local repository: `gin commit [--json] [--message message] [<filenames>]`
     - (note that you don’t have a way to stage/add individual files)?.
   - To upload the local changes to the remote repository on the GIN server: `gin upload`

### To update the data

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

- To get the most updated repository: `gin download`
- To sync the changes bi-directionally: `gin sync`

### To setup GIN repo in ceph

1. Mount ceph in computer connected to SWC network (see [[work journals/SWC Computing#Mounting ceph temporarily on stitch|SWC Computing]])
2. run `sudo gin init`
3. add exception for mounted directory
   `git config --global --add safe.directory /mnt/ceph-zoo-sminano/crabs_bboxes_labels/Aug2023_day3`
4. run `sudo gin add-remote origin gin:sfmig/crabs-ceph-aug2023-day3`

For troubleshooting:

- GIN error messages can be very succinct, to troubleshoot is often useful to treat the repo as a regular git repo (and try to push for example)?

Other useful workflows

- To delete a GIN repository but keeping git repo:
  - delete in the browser and then log back into gin?
  - `git annex uninit`
    - https://stackoverflow.com/questions/24447047/remove-git-annex-repository-from-file-tree
    - this removes relevant bits in .git/annex and .git/objects - some pre-commits may need to be edited by hand

To fetch the data in your code
...

To extract the checksum

> Determine the sha256 checksum hash of each new file, by running sha256sum <filename> in a terminal. Alternatively, you can use pooch to do this for you: python -c "import pooch; pooch.file_hash('/path/to/file')". If you wish to generate a text file containing the hashes of all the files in a given folder, you can use python -c "import pooch; pooch.make_registry('/path/to/folder', 'sha256_registry.txt').

"""A notebook to remove duplicates in groundtruth file and save the output"""

import ast
from pathlib import Path

import pandas as pd

# %%%%%%%%%%
# Enable interactive plots
# %matplotlib widget

# %%%%%%%%%%%%%%%%%%%%%
# load predictions
groundtruth_csv = (
    "/Users/sofia/arc/project_Zoo_crabs/escape_clips/"
    "04.09.2023-04-Right_RE_test_corrected_ST_csv_SM.csv"
)


# %%%%%%%%%%%%%%%%%%%%%%%%%
# Fix ground truth file and save

# read as a dataframe
df = pd.read_csv(groundtruth_csv, sep=",", header=0)

# find duplicates
list_unique_filenames = list(set(df.filename))
filenames_to_rep_ID = {}
for file in list_unique_filenames:
    df_one_filename = df.loc[df["filename"] == file]

    # track IDs for one file
    list_track_ids_one_filename = [
        int(ast.literal_eval(row.region_attributes)["track"])
        for row in df_one_filename.itertuples()
    ]

    # if duplicates: compute the repeated track IDs for this file
    # there could be more than one duplicate!!!
    if len(set(list_track_ids_one_filename)) != len(
        list_track_ids_one_filename
    ):
        # remove first occurrence of each track ID
        for k in set(list_track_ids_one_filename):
            list_track_ids_one_filename.remove(k)

        # the track IDs that remain are the repeated ones
        filenames_to_rep_ID[file] = list_track_ids_one_filename

# delete duplicate rows
for file, list_rep_ID in filenames_to_rep_ID.items():
    for rep_ID in list_rep_ID:
        # find repeated rows for selected file and rep_ID
        matching_rows = df[
            (df["filename"] == file)
            & (df["region_attributes"] == f'{{"track":"{rep_ID}"}}')
        ]

        # Identify the index of the first matching row
        if not matching_rows.empty:
            indices_to_drop = matching_rows.index[1:]

            # Drop all but the first matching row
            df = df.drop(indices_to_drop)

# save to csv
groundtruth_csv_corrected = Path(groundtruth_csv).parent / Path(
    Path(groundtruth_csv).stem + "_corrected.csv"
)
df.to_csv(groundtruth_csv_corrected, index=False)

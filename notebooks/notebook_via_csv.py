# %%
import csv
from pathlib import Path

# %%
tracking_output_dir = Path(".")
csv_file = open(str(tracking_output_dir / "tracking_output.csv"), "w")
csv_writer = csv.writer(
    csv_file,
    delimiter=",",
    quotechar="'",
    doublequote=True,
    skipinitialspace=False,
    lineterminator="\n",
    quoting=csv.QUOTE_MINIMAL,
    # dialect=csv.unix_dialect(),
    # delimiter=',',
    # quotechar='"',
    # doublequote=True,  #when writing, each quote character embedded in the data is written as two quotes
    # quoting=csv.QUOTE_MINIMAL,
)

# write header following VIA convention
# https://www.robots.ox.ac.uk/~vgg/software/via/docs/face_track_annotation.html
csv_writer.writerow(
    (
        "filename",
        "file_size",
        "file_attributes",
        "region_count",
        "region_id",
        "region_shape_attributes",
        "region_attributes",
    )
)
# %%
for frame_name in [f"{idx}" for idx in range(5)]:
    csv_writer.writerow(
        (
            frame_name,
            0,
            {"clip": 123},
            1,
            0,
            {
                "name": "rect",
                "x": 480,
                "y": 480,
                "width": 480,
                "height": 480,
            },
            {"track": 0},
        )
    )

# %%
csv_file.close()
# %%

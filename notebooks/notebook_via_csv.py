# %%
import csv

# %%

csv_file = open("tracking_output.csv", "w")
csv_writer = csv.writer(
    csv_file,
    delimiter=",",
    quotechar="'",  # interpret this as the quoting char
    doublequote=True,
    skipinitialspace=False,
    lineterminator="\n",
    quoting=csv.QUOTE_MINIMAL,  # when a field contains either the quotechar or the delimiter
    # https://github.com/python/cpython/blob/e7ba6e9dbe5433b4a0bcb0658da6a68197c28630/Lib/csv.py#L46
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

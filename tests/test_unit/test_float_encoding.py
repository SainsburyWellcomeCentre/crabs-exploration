from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from zarr.codecs import TransposeCodec

from crabs.mask.utils.float_encoding import float_var_encoding


@pytest.mark.parametrize(
    "dims, expected_order",
    [
        (("clip_id", "time", "individual"), (0, 2, 1)),
        (("clip_id", "time", "space", "individual"), (0, 2, 3, 1)),
        (("time", "individual"), (1, 0)),
    ],
)
def test_float_var_encoding_moves_time_last(dims, expected_order):
    """Check the transpose moves `time` last and keeps the others in order."""
    encoding = float_var_encoding(dims)
    assert encoding["filters"] == [TransposeCodec(order=expected_order)]


def test_float_var_encoding_is_lossless(tmp_path: Path):
    """Check values are read back bit-identical, NaN included."""
    rng = np.random.default_rng(0)
    values = rng.normal(size=(2, 50, 2, 7)).astype(np.float32)
    values[:, 10:30, :, 3] = np.nan  # an individual absent for a while
    values[0, 0, 0, 0] = -0.0
    ds = xr.Dataset(
        {"position": (("clip_id", "time", "space", "individual"), values)}
    )

    # Write with the float encoding, in chunks that do not divide the dims
    ds.to_zarr(
        tmp_path / "store.zarr",
        encoding={
            "position": {
                "chunks": (1, 16, 2, 4),
                **float_var_encoding(ds.position.dims),
            }
        },
    )

    # Check values are bit-identical
    read = xr.open_zarr(tmp_path / "store.zarr").position.values
    np.testing.assert_array_equal(read.view(np.uint32), values.view(np.uint32))

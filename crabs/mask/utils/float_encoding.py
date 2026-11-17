"""Zarr encoding of the float variables of a masks store.

The boxes and the mask and ellipse parameters are float32 arrays with a
`time` and an `individual` axis, and NaN wherever a crab has no box. Two
lossless codecs make them ~39% smaller on real data than zarr's default
(zstd alone), with bit-identical values:

- A transpose, so within each chunk the values of one individual over
  time are contiguous on disk, not the values of all individuals at one
  frame. A crab's values change little from frame to frame, and the NaN
  where it is absent become long runs.
- A byte shuffle (in blosc, with zstd inside), which groups the 1st,
  2nd, 3rd and 4th bytes of every float. Along a track the high bytes
  barely change, so they compress to almost nothing.

Both codecs are part of the zarr v3 spec and are undone on read, so the
arrays keep their dims and are read as before.
"""

from zarr.codecs import BloscCodec, TransposeCodec


def float_var_encoding(dims: tuple[str, ...]) -> dict:
    """Return the zarr encoding of a float variable with the given dims.

    Parameters
    ----------
    dims : tuple[str, ...]
        The variable's dims, which must include `time`.

    Returns
    -------
    dict
        The variable's `filters` and `compressors`, to pass in the
        `encoding` of `to_zarr`.

    """
    # Move `time` to the last axis on disk, keeping the others in order
    time_axis = dims.index("time")
    order = (*(i for i in range(len(dims)) if i != time_axis), time_axis)
    return {
        "filters": [TransposeCodec(order=order)],
        "compressors": [BloscCodec(cname="zstd", clevel=5, shuffle="shuffle")],
    }

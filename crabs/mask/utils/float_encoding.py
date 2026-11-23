"""Zarr encoding of the float variables of a masks store.

The boxes, the mask and ellipse parameters are float32 arrays with a
`time` and an `individual` axis, and NaN wherever there is no box. We use
two lossless codecs that make their size ~39% smaller than zarr's default
(zstd alone), with bit-identical values:

- A transpose, so within each chunk the values of one individual over
  time are contiguous on disk. Values change little from frame to frame,
  and the NaN where a crab is absent form long runs, so both compress
  better.

- A byte shuffle (in blosc, with zstd inside), which groups the 1st,
  2nd, 3rd and 4th bytes of every float. Along a trajectory, the high
  bytes barely change, so they are also amenable to compression.

Both codecs are undone on read, and the arrays keep their dims.
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

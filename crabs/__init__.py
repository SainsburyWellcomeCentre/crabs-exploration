from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("crabs")
except PackageNotFoundError:
    # package is not installed
    pass

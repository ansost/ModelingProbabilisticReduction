"""Train an NDL model.

Usage:
    python trainNDL.py
"""

from pyndl import ndl

if __name__ == "__main__":
    weights = ndl.ndl(
        events="../data/final_eventfile_buckeye.gz",
        alpha=0.1,
        betas=(0.1, 0.1),
        lambda_=1.0,
        method="openmp",
        remove_duplicates=True,
        verbose=True,
    )

    weights.to_netcdf("../data/weights_buckeye.nc")

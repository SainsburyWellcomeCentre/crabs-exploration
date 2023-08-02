# %%
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

# %%

plt.figure()

for train_subdir in ["crabs_pose_4k_TD2", "crabs_pose_4k_TD4"]:
    # file location
    train_job_dir = Path("/Volumes/zoo/users/sminano/" + train_subdir)
    model_subdir = Path(
        "labels.v001.slp.training_job/models/230725_174219.centered_instance"
    )  # 230728_180051
    train_log_csv = train_job_dir / model_subdir / "training_log.csv"

    # %
    # read file
    # train_log_array = np.genfromtxt(
    #     train_log_csv,
    #     delimiter=",", |
    #     filling_values=np.nan
    # )

    train_log_df = pd.read_csv(
        train_log_csv,
        index_col="epoch",
    )

    # plot
    # plt.figure()
    plt.plot(
        train_log_df.index,
        train_log_df["loss"],
        "--",
        label="TRAIN " + train_subdir + "_" + model_subdir.suffix[1:],
    )
    plt.plot(
        train_log_df.index,
        train_log_df["val_loss"],
        "-",
        label="VAL " + train_subdir + "_" + model_subdir.suffix[1:],
    )
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()


# %%

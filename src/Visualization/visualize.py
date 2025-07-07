import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from IPython.display import display

# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------

df = pd.read_pickle("../../data/interim/01_data_processed.pkl")

# --------------------------------------------------------------
# Plot single columns
# --------------------------------------------------------------

set_df = df[df["set"] == 1]

plt.plot(set_df["acc_y"])  # now compare with the df   # plt.plot(df["acc_x"])

plt.plot(set_df["acc_y"].reset_index(drop=True))

# Aswe are not understanding what it showes We plot as per label
# --------------------------------------------------------------
# Plot all exercises/labes
# --------------------------------------------------------------

for label in df["label"].unique():
    subset = df[df["label"] == label]
    # display(subset.head(2))
    plt.plot(
        subset["acc_y"].reset_index(drop=True), label=label
    )  # we working with all datapoints on acc_x
    plt.legend()
    plt.show()
    plt.close()

for label in df["label"].unique():
    subset = df[df["label"] == label]
    # display(subset.head(2))
    fig, ax = plt.subplots()
    plt.plot(
        subset[:100]["acc_y"].reset_index(drop=True), label=label
    )  # we added first 100 itteration of acc_x
    plt.legend()
    plt.show()
    plt.close()

# --------------------------------------------------------------
# Adjust plot settings
# --------------------------------------------------------------

mpl.style.use("seaborn-v0_8-deep")
mpl.rcParams["figure.figsize"] = [20, 5]
mpl.rcParams["figure.dpi"] = 100  # High Resolution for the plot

# --------------------------------------------------------------
# Compare medium vs. heavy sets
# --------------------------------------------------------------

category_df = df.query("label == 'squat'").query("participant == 'A'").reset_index()

fig, ax = plt.subplots()
category_df.groupby(["category"])["acc_y"].plot()
ax.set_ylabel("acc_y || Acceleration/ms^2")
ax.set_xlabel("Sample Veriation")
plt.legend()


# --------------------------------------------------------------
# Compare participants
# --------------------------------------------------------------

participant_df = df.query("label == 'bench'").sort_values("participant").reset_index()

fig, ax = plt.subplots()
participant_df.groupby(["participant"])["acc_y"].plot()
ax.set_ylabel("acc_y")
ax.set_xlabel("Sample Veriation")
plt.legend()


# --------------------------------------------------------------
# Plot multiple axis
# --------------------------------------------------------------

label = "squat"
participant = "A"

all_axis_df = (
    df.query(f"label == '{label}'")
    .query(f"participant == '{participant}'")
    .reset_index()
)

fig, ax = plt.subplots()
all_axis_df[["acc_x", "acc_y", "acc_z"]].plot(ax=ax)
ax.set_ylabel("acc_y")
ax.set_xlabel("Sample Veriation")
plt.legend()


# --------------------------------------------------------------
# Create a loop to plot all combinations per sensor
# --------------------------------------------------------------

label = df["label"].unique()
participant = df["participant"].unique()

for label in df["label"].unique():
    for participant in df["participant"].unique():
        all_axis_df = (
            df.query(f"label == '{label}'")
            .query(f"participant == '{participant}'")
            .reset_index()
        )
        if len(all_axis_df) > 0:
            fig, ax = plt.subplots()
            all_axis_df[["acc_x", "acc_y", "acc_z"]].plot(ax=ax)
            ax.set_ylabel("Acceleration/ms^2")
            ax.set_xlabel("Sample Veriation")
            plt.title(f"Label: {label} - Participant: {participant}")
            plt.legend()
            plt.show()
            plt.close()

for label in df["label"].unique():
    for participant in df["participant"].unique():
        all_axis_df = (
            df.query(f"label == '{label}'")
            .query(f"participant == '{participant}'")
            .reset_index()
        )
        if len(all_axis_df) > 0:
            fig, ax = plt.subplots()
            all_axis_df[["gyr_x", "gyr_y", "gyr_z"]].plot(ax=ax)
            ax.set_ylabel("Gyroscope/rad/s")
            ax.set_xlabel("Sample Veriation")
            plt.title(f"Label: {label} - Participant: {participant}")
            plt.legend()
            plt.show()
            plt.close()


# --------------------------------------------------------------
# Combine plots in one figure
# --------------------------------------------------------------

label = "row"
participant = "A"

combained_plot_df = (
    df.query(f"label == '{label}'")
    .query(f"participant == '{participant}'")
    .reset_index(drop=True)
)

fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(20, 10))
combained_plot_df[["acc_x", "acc_y", "acc_z"]].plot(ax=ax[0])
combained_plot_df[["gyr_x", "gyr_y", "gyr_z"]].plot(ax=ax[1])

ax[0].legend(
    loc="upper right", bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=True, shadow=True
)
ax[1].legend(
    loc="upper right", bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=True, shadow=True
)
ax[1].set_xlabel("Sample Veriation")

plt.legend()
plt.show()
plt.close()

# --------------------------------------------------------------
# Loop over all combinations and export for both sensors
# --------------------------------------------------------------

labels = df["label"].unique()
participants = df["participant"].unique()


for label in labels:
    for participant in participants:
        combained_plot_df = (
            df.query(f"label == '{label}'")
            .query(f"participant == '{participant}'")
            .reset_index()
        )
        if len(combained_plot_df) > 0:
            fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(20, 10))
            combained_plot_df[["acc_x", "acc_y", "acc_z"]].plot(ax=ax[0])
            combained_plot_df[["gyr_x", "gyr_y", "gyr_z"]].plot(ax=ax[1])

            ax[0].legend(
                loc="upper right",
                bbox_to_anchor=(0.5, 1.15),
                ncol=3,
                fancybox=True,
                shadow=True,
            )
            ax[1].legend(
                loc="upper right",
                bbox_to_anchor=(0.5, 1.15),
                ncol=3,
                fancybox=True,
                shadow=True,
            )
            ax[1].set_xlabel("Sample Veriation")

            plt.savefig(f"../../reports/figures/{label.title()}_{participant}.png")
            plt.close()
            plt.show()

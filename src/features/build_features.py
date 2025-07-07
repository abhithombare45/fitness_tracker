import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from DataTransformation import LowPassFilter, PrincipalComponentAnalysis
from TemporalAbstraction import NumericalAbstraction
from FrequencyAbstraction import FourierTransformation
from sklearn.cluster import KMeans
# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------

df = pd.read_pickle(
    "/Users/abhijeetthombare/ab_lib/Projects/fitness_tracker/data/interim/02_data_outliers_removed_chauvenets.pkl"
)

predictor_column = list(df.columns[:6])

plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["lines.linewidth"] = 2

# --------------------------------------------------------------
# Dealing with missing values (imputation)
# --------------------------------------------------------------
for col in predictor_column:
    df[col] = df[col].interpolate()

df.info()

# --------------------------------------------------------------
# Calculating set duration
# --------------------------------------------------------------

df[df["set"] == 25]["acc_y"].plot()
df[df["set"] == 50]["acc_y"].plot()

duration = df[df["set"] == 1].index[-1] - df[df["set"] == 1].index[0]

for s in df["set"].unique():
    start = df[df["set"] == s].index[0]
    stop = df[df["set"] == s].index[-1]

    duration = stop - start
    df.loc[(df["set"] == s), "duration"] = duration.seconds

duratoin_df = df.groupby("category")["duration"].mean()

duratoin_df.iloc[0] / 5
duratoin_df.iloc[1] / 10


# --------------------------------------------------------------
# Butterworth lowpass filter
# --------------------------------------------------------------

df_lowpass = df.copy()
LowPass = LowPassFilter()

fs = 1000 / 200
cutoff = 1.3

df_lowpass = LowPass.low_pass_filter(df_lowpass, "acc_y", fs, cutoff, order=5)

subset = df_lowpass[df_lowpass["set"] == 45]
print(subset["label"][0])
fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(20, 10))
ax[0].plot(subset["acc_y"].reset_index(drop=True), label="raw_data")
ax[1].plot(subset["acc_y_lowpass"].reset_index(drop=True), label="Butterworth Filter")
ax[0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), fancybox=True, shadow=True)
ax[1].legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), fancybox=True, shadow=True)


for col in predictor_column:
    df_lowpass = LowPass.low_pass_filter(df_lowpass, col, fs, cutoff, order=5)
    df_lowpass[col] = df_lowpass[col + "_lowpass"]
    del df_lowpass[col + "_lowpass"]


# --------------------------------------------------------------
# Principal component analysis PCA
# --------------------------------------------------------------
df_pca = df_lowpass.copy()

PCA = PrincipalComponentAnalysis()

pc_values = PCA.determine_pc_explained_variance(df_pca, predictor_column)

# The Elbow Technioque Explanation
plt.figure(figsize=(10, 10))
plt.plot(range(1, len(predictor_column) + 1), pc_values)
plt.xlabel("principal component number")
plt.ylabel("explained variance")
plt.show()

df_pca = PCA.apply_pca(df_pca, predictor_column, 3)
subset = df_pca[df_pca["set"] == 45]
subset[["pca_1", "pca_2", "pca_3"]].plot()

# --------------------------------------------------------------
# Sum of squares attributes
# --------------------------------------------------------------

df_squared = df_pca.copy()

acc_r = df_squared["acc_x"] ** 2 + df_squared["acc_y"] ** 2 + df_squared["acc_z"] ** 2
gyr_r = df_squared["gyr_x"] ** 2 + df_squared["gyr_y"] ** 2 + df_squared["gyr_z"] ** 2

df_squared["acc_r"] = np.sqrt(acc_r)
df_squared["gyr_r"] = np.sqrt(gyr_r)

subset = df_squared[df_squared["set"] == 14]
subset[["acc_r", "gyr_r"]].plot(subplots=True)

df_squared

# --------------------------------------------------------------
# Temporal abstraction
# --------------------------------------------------------------

# If we check Class and what veriable we need to pass in to the functions,
# then we are good to declare veriable and work on creating code

df_temporal = df_squared.copy()

NumAbs = NumericalAbstraction()

predictor_column = predictor_column + ["acc_r", "gyr_r"]

ws = int(1000 / 200)

# for mean and Std deviation
for col in predictor_column:
    df_temporal = NumAbs.abstract_numerical(df_temporal, [col], ws, "mean")
    df_temporal = NumAbs.abstract_numerical(df_temporal, [col], ws, "std")


df_temporal_list = []

for s in df_temporal["set"].unique():
    subset = df_temporal[df_temporal["set"] == s].copy()
    for col in predictor_column:
        subset = NumAbs.abstract_numerical(subset, [col], ws, "mean")
        subset = NumAbs.abstract_numerical(subset, [col], ws, "std")
    df_temporal_list.append(subset)

df_temporal = pd.concat(df_temporal_list)

df_temporal.info()

subset[["acc_y", "acc_y_temp_mean_ws_5", "acc_y_temp_std_ws_5"]].plot()
subset[["gyr_y", "gyr_y_temp_mean_ws_5", "gyr_y_temp_std_ws_5"]].plot()


# --------------------------------------------------------------
# Frequency features
# --------------------------------------------------------------

df_freq = df_temporal.copy().reset_index()
FreqAbs = FourierTransformation()

fs = int(1000 / 200)
ws = int(2800 / 200)

df_freq = FreqAbs.abstract_frequency(df_freq, ["acc_y"], ws, fs)


# Visualize result
subset = df_freq[df_freq["set"] == 15]
subset[["acc_y"]].plot()
subset[
    [
        "acc_y_max_freq",
        "acc_y_freq_weighted",
        "acc_y_pse",
        "acc_y_freq_1.429_Hz_ws_14",
        "acc_y_freq_2.5_Hz_ws_14",
    ]
].plot()

# applying on all the column
df_freq_list = []
for s in df_freq["set"].unique():
    print(f"Applying Fourier transformations to set {s}")
    subset = df_freq[df_freq["set"] == s].reset_index(drop=True).copy()
    subset = FreqAbs.abstract_frequency(subset, predictor_column, ws, fs)
    df_freq_list.append(subset)


df_freq = pd.concat(df_freq_list)

df_freq = pd.concat(df_freq_list).set_index("epoch (ms)", drop=True)

# --------------------------------------------------------------
# Dealing with overlapping windows
# --------------------------------------------------------------

df_freq = df_freq.dropna()

# below RHS is Identical DF to LHS df
""" df_freq == df_freq.iloc[:,:] """

# # applying 50% metod to reduce data
df_freq.iloc[::2]
""" You can see every 2nd ros is removed like e.i.,'wise  200 ms row excluded then 600 ms as so on"""
"""this will sort out overfitting issue from the data."""

# making df
df_freq = df_freq.iloc[::2]

# --------------------------------------------------------------
# Clustering
# --------------------------------------------------------------

df_cluster = df_freq.copy()

cluster_column = ["acc_x", "acc_y", "acc_z"]
k_values = range(2, 10)
inertias = []  # initially it sould by empty

for k in k_values:
    subset = df_cluster[cluster_column]
    kmeans = KMeans(n_clusters=k, n_init=20, random_state=0)
    cluster_labels = kmeans.fit_predict(subset)
    inertias.append(kmeans.inertia_)

plt.figure(figsize=(10, 10))
plt.plot(k_values, inertias)
plt.xlabel("k")
plt.ylabel("Sum of squared distances")
plt.show()

kmeans = KMeans(n_clusters=5, n_init=20, random_state=0)
subset = df_cluster[cluster_column]
df_cluster["cluster"] = kmeans.fit_predict(subset)

# Cluster Plot
fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(projection="3d")
for c in df_cluster["cluster"].unique():
    subset = df_cluster[df_cluster["cluster"] == c]
    ax.scatter(subset["acc_x"], subset["acc_y"], subset["acc_z"], label=c)
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")
plt.legend()
plt.show()

# with Label
fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(projection="3d")
for l in df_cluster["label"].unique():
    subset = df_cluster[df_cluster["label"] == l]
    ax.scatter(subset["acc_x"], subset["acc_y"], subset["acc_z"], label=l)
ax.set_xlabel("X-Axis")
ax.set_xlabel("Y-Axis")
ax.set_xlabel("Z-Axis")
plt.legend()
plt.show()

# ===========================
# Code from CHATGPT for plot
# ===========================

# import matplotlib.pyplot as plt
# import seaborn as sns

# # Define a stylish color palette
# colors = sns.color_palette("husl", n_colors=len(df_cluster["cluster"].unique()))

# # Create a fancy 3D plot
# fig = plt.figure(figsize=(15, 15))
# ax = fig.add_subplot(projection="3d", facecolor="black")

# # Loop through unique clusters
# for idx, c in enumerate(df_cluster["cluster"].unique()):
#     subset = df_cluster[df_cluster["cluster"] == c]
#     ax.scatter(
#         subset["acc_x"], subset["acc_y"], subset["acc_z"],
#         label=c, color=colors[idx], s=60, alpha=0.85, edgecolors="white"
#     )

# # Set labels with stylish fonts
# ax.set_xlabel("X-axis", fontsize=14, fontweight="bold", color="white")
# ax.set_ylabel("Y-axis", fontsize=14, fontweight="bold", color="white")
# ax.set_zlabel("Z-axis", fontsize=14, fontweight="bold", color="white")
# ax.set_title("3D Cluster Visualization", fontsize=16, fontweight="bold", color="white")

# # Customize grid and legend
# ax.grid(color="gray", linestyle="dashed", linewidth=0.5, alpha=0.5)
# ax.legend(fontsize=12, loc="upper right", frameon=True, facecolor="black", edgecolor="white", labelcolor="white")
# # Rotate for better visibility
# ax.view_init(elev=20, azim=45)
# # Show the plot
# plt.show()


# #===============================================================
# # Animation :
# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np
# import matplotlib.animation as animation

# # Define a color palette
# colors = sns.color_palette("plasma", as_cmap=True)

# # Create figure
# fig = plt.figure(figsize=(12, 10))
# ax = fig.add_subplot(projection="3d", facecolor="black")

# # Normalize cluster values for gradient effect
# unique_clusters = df_cluster["cluster"].unique()
# norm = plt.Normalize(vmin=unique_clusters.min(), vmax=unique_clusters.max())

# # Loop through unique clusters
# for c in unique_clusters:
#     subset = df_cluster[df_cluster["cluster"] == c]
#     sizes = np.linspace(50, 120, len(subset))  # Vary marker sizes for depth

#     scatter = ax.scatter(
#         subset["acc_x"], subset["acc_y"], subset["acc_z"],
#         c=[c] * len(subset), cmap=colors, norm=norm,
#         s=sizes, alpha=0.85, edgecolors="white", linewidth=0.6
#     )

# # Labels and title
# ax.set_xlabel("X-axis", fontsize=14, fontweight="bold", color="white")
# ax.set_ylabel("Y-axis", fontsize=14, fontweight="bold", color="white")
# ax.set_zlabel("Z-axis", fontsize=14, fontweight="bold", color="white")
# ax.set_title("3D Cluster Visualization with Auto-Rotation", fontsize=16, fontweight="bold", color="white")

# # Grid and colorbar
# ax.grid(color="gray", linestyle="dashed", linewidth=0.5, alpha=0.5)
# cbar = fig.colorbar(scatter, ax=ax, pad=0.1, shrink=0.6)
# cbar.set_label("Cluster Index", fontsize=12, color="white")
# plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color="white")

# # Function to update rotation
# def rotate(angle):
#     ax.view_init(elev=20, azim=angle)

# # Animate the rotation
# ani = animation.FuncAnimation(fig, rotate, frames=np.arange(0, 360, 2), interval=100)

# plt.show()

# ===============================
# END Code from CHATGPT for plot
# ===============================


# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------

df_cluster.to_pickle("../../data/interim/03_data_features.pkl")

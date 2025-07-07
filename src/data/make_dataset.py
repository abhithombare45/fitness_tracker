import pandas as pd
from glob import glob

# --------------------------------------------------------------
# Read single CSV file
# --------------------------------------------------------------

# Read single file
single_file_acc = pd.read_csv(
    "/Users/abhijeetthombare/ab_lib/Projects/fitness_tracker/data/raw/MetaMotion/A-bench-heavy2-rpe8_MetaWear_2019-01-11T16.10.08.270_C42732BE255C_Accelerometer_12.500Hz_1.4.4.csv"
)

single_file_gyr = pd.read_csv(
    "/Users/abhijeetthombare/ab_lib/Projects/fitness_tracker/data/raw/MetaMotion/A-bench-heavy2-rpe8_MetaWear_2019-01-11T16.10.08.270_C42732BE255C_Gyroscope_25.000Hz_1.4.4.csv"
)


# --------------------------------------------------------------
# List all data in data/raw/MetaMotion
# --------------------------------------------------------------

files = glob(
    "/Users/abhijeetthombare/ab_lib/Projects/fitness_tracker/data/raw/MetaMotion/*.csv"
)

len(files)


# --------------------------------------------------------------
# Extract features from filename
# --------------------------------------------------------------

data_path = (
    "/Users/abhijeetthombare/ab_lib/Projects/fitness_tracker/data/raw/MetaMotion/"
)


# Finding Participant from the data
# f.split("-")
# f.split("-")[0]

f = files[0] # take single element from array to work on whole dataframe elements.

participant = f.split("-")[0].replace(data_path, "")
label = f.split("-")[1]
category = f.split("-")[2].rstrip("123")

# Creating new dataset with extracted fields like participant, label, category

df = pd.read_csv(f)  # df is new data set with all the files aat the location

df["participant"] = participant
df["label"] = label
df["category"] = category

# --------------------------------------------------------------
# Read all files
# --------------------------------------------------------------




 acc_df[acc_df["set"] == 1]   

# --------------------------------------------------------------
# Working with datetimes
# --------------------------------------------------------------

acc_df.info()

df["epoch (ms)"]
pd.to_datetime(df["epoch (ms)"], unit="ms")

df["time (01:00)"] # ignoring epoch (ms) column data and time (01:00) column data similarity as its taken in daylight situation.
    #above is type object and below is type datetime64[ns]
pd.to_datetime(df["time (01:00)"])

pd.to_datetime(df["time (01:00)"]).dt.month

acc_df.index #normally, this is range index 



#now we have clean data frame with timestamp as index
acc_df
gyr_df


# --------------------------------------------------------------
# Turn into function
# --------------------------------------------------------------


files = glob(
    "/Users/abhijeetthombare/ab_lib/Projects/fitness_tracker/data/raw/MetaMotion/*.csv"
)

data_path = (
    "/Users/abhijeetthombare/ab_lib/Projects/fitness_tracker/data/raw/MetaMotion/"
)

def read_data_from_files(files):
    acc_df = pd.DataFrame()
    gyr_df = pd.DataFrame()

    acc_set = 1
    gyr_set = 1

    for f in files:
        participant = f.split("-")[0].replace(data_path, "")
        label = f.split("-")[1]
        category = (f.split("-")[2].rstrip("123").rstrip("_MetaWear_2019"))  # removing "_MetaWear_2019"

        # print(participant, label, category)

        df = pd.read_csv(f)

        df["participant"] = participant
        df["label"] = label
        df["category"] = category

        if "Accelerometer" in f:
            df["set"] = acc_set
            acc_set += 1
            acc_df = pd.concat([acc_df, df])

        if "Gyroscope" in f:
            df["set"] = gyr_set
            gyr_set += 1
            gyr_df = pd.concat([gyr_df, df])
    
    acc_df.index = pd.to_datetime(acc_df["epoch (ms)"], unit="ms") #changing index to datetime for acc data
    gyr_df.index = pd.to_datetime(gyr_df["epoch (ms)"], unit="ms")  #changing index to datetime for gyr data

    del acc_df["epoch (ms)"]
    del acc_df["time (01:00)"]
    del acc_df["elapsed (s)"]

    del gyr_df["epoch (ms)"]
    del gyr_df["time (01:00)"]
    del gyr_df["elapsed (s)"]

    return acc_df, gyr_df


acc_df, gyr_df = read_data_from_files(files)


# --------------------------------------------------------------
# Merging datasets
# --------------------------------------------------------------

data_merged = pd.concat([acc_df.iloc[:,:3], gyr_df], axis=1) #approx. 70k tuples

# observation for gather best results
# 
data_merged.head(45) # what we observed that gyroscope data is too large as it gathered large dataset
data_merged.dropna() # this will remove all the rows with NaN values and combine data of both acc and gyr

# RENAMEING COLUMNS FOR BETER UNDERSTANDING
data_merged.columns = [
    "acc_x",
    "acc_y",
    "acc_z",
    "gyr_x",
    "gyr_y",
    "gyr_z",
    "participant",
    "label",
    "category",
    "set",
]

# --------------------------------------------------------------
# Resample data (frequency conversion)
# --------------------------------------------------------------

# Accelerometer:    12.500HZ
# Gyroscope:        25.000Hz

#  1/25 = 0.04 seconds # Gyroscope data is gathered after every 0.04 seconds
#  1/12.5 = 0.08 seconds # Accelerometer data is gathered after every 0.08 seconds

# data_merged[:100].resample(rule="S") this is nothing but a function to resample the data
# this will take mean of all the values in the data and resample it to 1 second
data_merged[:100].resample(rule="S") 

# If there are non-numeric columns in data_merge, exclude them before resampling:
# numeric_data = data_merged.select_dtypes(include=['number'])
# resampled_data = numeric_data[:1000].resample(rule="200ms").apply(sampling)
# print(resampled_data)
# print()

data_merged[:100].resample(rule="S").mean()

sampling = {
   'acc_x': "mean", 
   'acc_y': "mean", 
   'acc_z': "mean", 
   'gyr_x': "mean", 
   'gyr_y': "mean", 
   'gyr_z': "mean", 
   'participant': "last",
   'label': "last",
   'category': "last",
   'set': "last",
}

data_merged.columns  # to create above sampling field we need to know the columns in the data_merged

data_merged[:1000].resample(rule="200ms").apply(sampling)


# If we go for whole dataset this will utilize more memory and time
# Data is ather for week and we can split it by day wise.

# split by day
days = [g for n, g in data_merged.groupby(pd.Grouper(freq="D"))]


days[0] # first day
days[1] # second day
days[-1] # last day

#--------------------------------------------------------------
# Resample data by day, this will help us save time and memory.
#--------------------------------------------------------------

data_resampled = pd.concat([df.resample(rule="200ms").apply(sampling).dropna() for df in days])
data_resampled["set"] = data_resampled["set"].astype("int")

data_resampled.info()



# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------


data_resampled.to_pickle("/Users/abhijeetthombare/ab_lib/Projects/fitness_tracker/data/interim/01_data_processed.pkl")
import argparse
from pathlib import Path
import os
import pandas as pd
import pickle
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
import sys
import timeit
import numpy as np

parser=argparse.ArgumentParser("prep")
parser.add_argument("--input_data", type=str, help="Name of the Input data file")
parser.add_argument("--output_data_train", type=str, help="Name of folder we will write training results out to")
parser.add_argument("--output_data_test", type=str, help="Name of folder we will write test results out to")

args=parser.parse_args()

print("Performing Data Cleaning & Feature Engineering.....")
print("Testing Workflow")

lines=[
    f"Input data path: {args.input_data}",
    f"Output training data path: {args.output_data_train}",
    f"Output test data path: {args.output_data_test}",
]

for line in lines:
    print(line)

df = pd.read_csv(Path(args.input_data), delimiter = ',', encoding = 'utf-8')

# We need to find and replace missing values for police district
# If a value is missing, replace it with 0.
print("Replacing missing police districts...")
df['Police_District'].fillna(0, inplace=True)

# Feature engineering steps
# year of issuance
df['Issued_date']=pd.to_datetime(df['Issued_date'])
df['Issued_year']=df['Issued_date'].dt.year

# categorize time based on hour of the day
# NOTE: cuts are right-aligned, so I'm starting with -1 to get the 0-6 hour range
hour_bins=[-1, 6, 10, 16, 19, np.inf ]
hour_names=['Overnight', 'Morning', 'Midday', 'AfterWork', 'Evening']
df['Time_of_day']=pd.cut(df['Issued_date'].dt.hour, bins=hour_bins, labels=hour_names)

# license plate origin
conds=[
    df['License_Plate_State'].isin(['IL']),
    df['License_Plate_State'].isin(['ON', 'ZZ', 'NB', 'AB', 'QU', 'MX', 'BC', 'MB', 'PE', 'NS', 'PQ', 'NF'])
]
choices=['In-state', 'Out-of-country']
df['License_plate_origin']=np.select(conds, choices, default='Out-of-state')

# vehicle type
conds=[
    df['Plate_Type'] == 'PAS',
    df['Plate_Type'] == 'TRK',
    df['Plate_Type'] == 'TMP'
]
choices=['PAS', 'TRK', 'TMP']
df['Vehicle_type']=np.select(conds, choices, default='Other')

# Feature selection
df.drop(['Tract', 'Hardship_Index', 'Issued_date', 'License_Plate_State', 'Plate_Type'], axis=1, inplace=True)

# Split the data into training and test sets
train_df, test_df=train_test_split(df, test_size=0.2, random_state=11084)

# Write the results out for the next step.
print("Writing training dataset to:", str(Path(args.output_data_train) / "TrainData.csv"))
train_df.to_csv((Path(args.output_data_train) / "TrainData.csv"), index=False)

print("Writing test dataset to:", str(Path(args.output_data_test) / "TestData.csv"))
test_df.to_csv((Path(args.output_data_test) / "TestData.csv"), index=False)

print("Done!")
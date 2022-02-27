import numpy as np
import pandas as pd
import glob

path = r'trainingData' # use your path
all_files = glob.glob(path + "/*.csv")

final_df = []
selectedHandPoints = [0,4,8,20]
frameLimit = 3

column_names = ["Label"]
for i in range(frameLimit):
    column_names.append("size_ratio_"+str(i))
    for j in selectedHandPoints:
        column_names.append("dist_"+str(j)+"_"+str(i))
        column_names.append("angle_"+str(j)+"_"+str(i))
        column_names.append("distC_"+str(j)+"_"+str(i))
        column_names.append("angleC_"+str(j)+"_"+str(i))

for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    label = df['Label'][0]
    df = df.iloc[: , 2:]
    full_row = [label]
    for index, row in df.iterrows():
        full_row = full_row + list(row)
        if index % frameLimit == frameLimit-1:
            final_df.append(full_row)
            full_row = [label]

frame = pd.DataFrame(final_df, columns=column_names)
frame.to_csv("trainingData.csv")

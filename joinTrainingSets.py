from matplotlib.pyplot import axis
import numpy as np
import pandas as pd
import glob

path = r'trainingData' # use your path
all_files = glob.glob(path + "/*.feather")

final_df = []
selectedHandPoints = [0,4,8,20]
frameLimit = 5

column_names = ["Label"]
for i in range(frameLimit):
    column_names.append("size_ratio_"+str(i))
    for j in selectedHandPoints:
        column_names.append("dist_"+str(j)+"_"+str(i))
        column_names.append("angle_"+str(j)+"_"+str(i))
        column_names.append("distC_"+str(j)+"_"+str(i))
        column_names.append("angleC_"+str(j)+"_"+str(i))

for filename in all_files:
    df = pd.read_feather(filename)
    label = df['Label'][0]
    print(df)
    df = df.drop('Label', axis=1)
    print(df)
    full_row = [label]
    for index, row in df.iterrows():
        full_row = full_row + list(row)
        if index % frameLimit == frameLimit-1:
            final_df.append(full_row)
            full_row = [label]

frame = pd.DataFrame(final_df, columns=column_names)
print(frame.shape)
frame.to_feather("trainingData.feather")

"""
Author: Luis Ponce Pacheco
Contact: luis.poncepacheco@wur.nl
PSG, ABE group.
"""

import mesa
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data_df = pd.read_csv("RobotCow_Data.csv")
print(data_df.keys())

data_df_filtered = data_df[(data_df.recruit_prob == 1) & (data_df.memory_threshold == 10)]  # 40
print(data_df_filtered.head(13))

# Create a lineplot with error bars
g = sns.lineplot(
    data=data_df_filtered,
    x="Step",
    y="Manure",
    hue="robot_num",
    errorbar=("ci", 95),
    palette="tab10",
)
g.figure.set_size_inches(8, 4)
plot_title = "recruit_prob = 1, memory_threshold = 10"
g.set(title=plot_title, xlabel="Time (s)")

plt.show()

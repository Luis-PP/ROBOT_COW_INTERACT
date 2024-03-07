import mesa
import pandas as pd
import numpy as np
from robot_cow_interact.model import RobotCow

# parameter lists for each parameter to be tested in batch run
parameters = {
    "width": 1101,
    "height": 501,
    "cow_num": 28,
    "robot_num": range(1, 15, 2),
    "recruit_prob": list(np.linspace(0, 1, 6)),
    "memory_threshold": range(10, 50, 10),
}

if __name__ == "__main__":
    data = mesa.batch_run(
        RobotCow,
        parameters=parameters,
        number_processes=8,
        max_steps= 10000,
        data_collection_period=1,
        display_progress=True,
    )
    data_df = pd.DataFrame(data)
    data_df.to_csv("RobotCow_Data.csv")

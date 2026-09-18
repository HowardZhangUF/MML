import pandas as pd
import os


def get_history_dataframe(dataframe, offset, history_length, if_time=False):
    """Get the history of the dataframe. The history is the last N_th rows of the dataframe.
    The history is flattened and concatenated to the final dataframe.

    Args:
        dataframe (pd.DataFrame): input dataframe
        offset (int): offset to start the history
        history_length (int): length of the history
        if_time (bool): if True, embed relative time encoding in the feature vector
    """
    final_df = pd.DataFrame()

    for i in range(0 + offset, len(dataframe) - history_length):
        non_flattened_i_hist = dataframe.iloc[i : i + history_length + 1].copy()
        if if_time:
            # for every value in rosbagTimestamp column, calculate the difference between the last value in the row (-1) and the current value
            non_flattened_i_hist.loc[:, "rosbagTimestamp"] = (
                non_flattened_i_hist["rosbagTimestamp"].iloc[-1]
                - non_flattened_i_hist["rosbagTimestamp"]
            )
        i_history = non_flattened_i_hist.values.flatten()
        final_df = pd.concat([final_df, pd.DataFrame(i_history).T], ignore_index=True)
    return final_df


def create_dataset():

    is_veloctities = True  # output velocities if true, else output positions
    is_time = True  # embed relative time encoding in the feature vector 
    is_occlusions = True  # remove positions in occlusions zones if true
    is_perfect = False  # use perfect measurements if true, else use noisy measurements

    is_connected_graph = False
    if is_connected_graph:
        file = (
            "connected_graph_training_2024-12-13-15-51-30__slash_noisy_measurement.csv"
        )
    else:
        topic_name = "robot0_slash_odom" if is_perfect else "noisy_measurement"
        file = "train_RNv2_2025-02-23-21-20-10__slash_" + topic_name + ".csv"

    N_th = 5  # Number of time steps to consider for the history

    path = os.path.expanduser("~/mml_ws/src/mml_guidance/sim_data/training_data/")
    df = pd.read_csv(
        path + file,
    )

    # get only x and y positions
    if is_perfect:
        reduced_df = df.iloc[::10, 11:13]
    else:
        reduced_df = df.iloc[::, 8:10] 

    print("first three rows: \n", reduced_df.head(3))
    print("length of df: ", len(reduced_df))

    if is_occlusions:
        occ_widths = [0.6, 0.6, 0.9, 0.6]
        occ_centers = [[-1.5, 1], [-0.75, 0.], [1.1, 1.1], [1.7, 0.]]
        occlusions = [
            [
                occ_centers[i][0] - occ_widths[i] / 2,
                occ_centers[i][0] + occ_widths[i] / 2,
                occ_centers[i][1] - occ_widths[i] / 2,
                occ_centers[i][1] + occ_widths[i] / 2,
            ]
            for i in range(len(occ_centers))
        ]  # [x_min, x_max, y_min, y_max]

        # remove rows where the x and y values are within the occlusion range
        for occ in occlusions:
            reduced_df = reduced_df[
                ~(
                    (reduced_df.iloc[:, 0] >= occ[0])
                    & (reduced_df.iloc[:, 0] <= occ[1])
                    & (reduced_df.iloc[:, 1] >= occ[2])
                    & (reduced_df.iloc[:, 1] <= occ[3])
                )
            ]
        print("length of df after occlusion removal: ", len(reduced_df))

    if is_time:
        df_time = df.iloc[reduced_df.index, 0] * 1e-9
        print("length of df_time: ", len(df_time))
        print("total time [min]: ", (df_time.iloc[-1] - df_time.iloc[0]) / 60)

        # get position dataframe with histories
        assert len(df_time) == len(reduced_df)
        dataframe = pd.concat([reduced_df, df_time], axis=1, ignore_index=True)

        dataframe.columns = ["x", "y", "rosbagTimestamp"]
    else:
        dataframe = reduced_df
        assert len(dataframe) == len(reduced_df)
        dataframe.columns = ["x", "y"]

    print("dataframe: \n", dataframe.head(3))
    if is_occlusions:
        # samples frac of the data to generate spots data so network learns about big jumps not only from occlusions
        frac = 0.7
        seed = 42
        sampled_df = dataframe.sample(frac=frac, replace=False, random_state=seed).sort_index()
        print("ordered sampled dataframe: \n", sampled_df.head(3))
        print("ordered sampled dataframe shape: \n", sampled_df.shape)
        final_df = get_history_dataframe(
            dataframe.sample(frac=frac, replace=False).sort_index(),
            offset=0,
            history_length=N_th,
            if_time=is_time,
        )
    else:
        final_df = get_history_dataframe(dataframe, offset=0, history_length=N_th, if_time=is_time)

    print("shape final df: ", final_df.shape)
    print("final df: \n", final_df.head(3))
    if is_veloctities:
        if is_time:
            # Calculate the velocities by backward differentiation
            final_df.iloc[:, -3:-1] = (
                final_df.iloc[:, -3:-1].values - final_df.iloc[:, -6:-4].values
            ) / final_df.iloc[:, -4].values.reshape(-1, 1)
            print("final df after adding velocities: \n", final_df.head(3))
        else:
            # Calculate the velocities by backward differentiation
            final_df.iloc[:, -2:] = (
                final_df.iloc[:, -2:].values - final_df.iloc[:, -4:-2].values
            ) / (1/3)
            print("final df after adding velocities: \n", final_df.head(3))

    # Save the final DataFrame to a new CSV file
    prefix_name = "noisy_" if not is_perfect else "perfect_"
    prefix_name = prefix_name + str(N_th) + "_v2_"

    if is_time:
        prefix_name += "time_"
    if is_veloctities:
        prefix_name += "velocities_"
    else:
        prefix_name += "position_"

    if is_occlusions:
        prefix_name += "occlusions_"

    if is_connected_graph:
        prefix_name = "connected_graph_" + prefix_name

    out_name = "converted_" + prefix_name + "training_data.csv"
    final_df.to_csv(path + out_name, index=False)
    print("Converted dataset saved to : ", path + out_name)


if __name__ == "__main__":
    create_dataset()

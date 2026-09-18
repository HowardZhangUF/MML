import os
import rospkg
import pandas as pd
import numpy as np

# Set the name of the input CSV directory
rospack = rospkg.RosPack()
package_dir = rospack.get_path("mml_guidance")
folder_path = package_dir + "/sim_data/processed_iros"


def main():
    outdir_all = folder_path + "/joined/summary/"
    if not os.path.exists(outdir_all):
        os.mkdir(outdir_all)

    include_data = {
        "err estimation norm": "$ee$ $[m]$",
        "err tracking norm": "$te$ $[m]$",
        "entropy data": "Entropy",
        "n eff particles data": "Effective Number of Particles",
        "xyTh estimate cov det": r"$\det({\Sigma_k})$  $[m^2]$",
        "is update data": r"\% FOV",
    }

    # empty dictionary to store the RMS values
    csv_dict = {}

    filenames = os.listdir(folder_path + "/joined/")
    print("filenames: ", filenames)
    for filename in filenames:
        if filename.endswith(".csv"):
            first_word = "_".join(filename.split("_")[:2])
            file_df = pd.read_csv(folder_path + "/joined/" + filename, low_memory=False)

            if "xyTh estimate cov det data" in file_df.columns:
                file_df = file_df.drop(columns=["xyTh estimate cov det"])
                file_df = file_df.rename(
                    columns={"xyTh estimate cov det data": "xyTh estimate cov det"}
                )

            # Collect data for RMS values csv
            csv_dict[first_word] = {}
            for col_name in file_df.columns:
                if (
                    any(col_name == word for word in include_data)
                    and col_name != "is update data"
                ):
                    rms = round(np.sqrt(np.mean(file_df[col_name] ** 2)), 3)
                    sd = round(np.std(file_df[col_name]), 3)  # standard deviation
                    csv_dict[first_word][col_name] = (rms, sd)
                elif col_name == "is update data":
                    # split the data file_df[col_name].dropna() into three equal parts
                    chunks = np.array_split(file_df[col_name].dropna(), 3)
                    # percent of the total length that 'is update data' column is true
                    perc = np.array(
                        [
                            round(100 * np.sum(chunk) / chunk.shape[0], 3)
                            for chunk in chunks
                        ]
                    )
                    csv_dict[first_word][col_name] = perc.mean()

    csv_df = pd.DataFrame.from_dict(csv_dict, orient="index")
    csv_df.T.to_csv(outdir_all + "all_runs_rms.csv", index=True)
    print("RMS values and Standard Deviation: \n", csv_df.T)


if __name__ == "__main__":
    main()

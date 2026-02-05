import os
import numpy as np
import pandas as pd
from pathlib import Path


def sem(x):
    return x.std(ddof=1) / np.sqrt(len(x))


def process_csv(input_file, base_output_dir, algorithms_to_use, col_var, x_var):

    print(f"\nProcessing {input_file}")

    df = pd.read_csv(input_file)

    # Filter requested algorithms
    df = df[df["algorithm"].isin(algorithms_to_use)]
    if df.empty:
        print("  WARNING: No matching algorithms — skipping.")
        return

    metrics = [
        "err_true",
        "err_central",
        "Log-likelihood"
    ]

    # Output folder for this CSV
    stem = Path(input_file).stem
    out_dir = Path(base_output_dir) / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    # Iterate per N_i
    for var, df_col_var in df.groupby(col_var):
        x_values = sorted(df_col_var[x_var].unique())

        for m in metrics:
            all_records = []
            for v in x_values:
                row_dict = {x_var: v}
                df_sub = df_col_var[df_col_var[x_var] == v]

                # compute mean + sem per algorithm
                for algo in algorithms_to_use:
                    df_algo = df_sub[df_sub["algorithm"] == algo]

                    if df_algo.empty or m not in df_algo.columns:
                        row_dict[f"{algo}_mean"] = np.nan
                        row_dict[f"{algo}_sem"] = np.nan
                    else:
                        row_dict[f"{algo}_mean"] = df_algo[m].mean()
                        row_dict[f"{algo}_sem"] = sem(df_algo[m])

                all_records.append(row_dict)

            # Convert table to DataFrame
            out_df = pd.DataFrame(all_records).sort_values(x_var)

            # Save CSV: e.g., NMI_Ni_10.csv
            metric_clean = m.replace(" ", "_")
            var_name = col_var.replace("_", "")
            f_name = f"{metric_clean}_"+var_name+f"_{var}.csv"
            out_file = out_dir / f_name
            out_df.to_csv(out_file, index=False, header=False, encoding="utf-8")
            print(f"    Saved → {out_file}")


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":

    base_results_dir = "processed_results"
    os.makedirs(base_results_dir, exist_ok=True)

    algorithms_to_use = ["central", "local", "fl"]

    csv_files = [
        # "df_iid.csv"
    ]

    for csv_path in csv_files:
        process_csv(csv_path, base_results_dir, algorithms_to_use, col_var="lambda", x_var="N")

    csv_files = [
        "df_lskew.csv"
    ]

    for csv_path in csv_files:
        process_csv(csv_path, base_results_dir, algorithms_to_use, col_var="alpha", x_var="N")

    # csv_files = [
    #     "df_data_sep.csv"
    # ]

    # for csv_path in csv_files:
    #     process_csv(csv_path, base_results_dir, algorithms_to_use, col_var="cluster_separation", x_var="local_spread")


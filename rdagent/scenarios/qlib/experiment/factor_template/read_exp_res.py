import os
import pickle
from pathlib import Path

import pandas as pd
import qlib
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
from model_health import best_iteration, ranking_stats

qlib.init()

from qlib.workflow import R

# here is the documents of the https://qlib.readthedocs.io/en/latest/component/recorder.html

# TODO: list all the recorder and metrics

# Assuming you have already listed the experiments
experiments = R.list_experiments()

# Iterate through each experiment to find the latest recorder
experiment_name = None
latest_recorder = None
for experiment in experiments:
    recorders = R.list_recorders(experiment_name=experiment)
    for recorder_id in recorders:
        if recorder_id is not None:
            experiment_name = experiment
            recorder = R.get_recorder(recorder_id=recorder_id, experiment_name=experiment)
            end_time = recorder.info["end_time"]
            try:
                # Check if the recorder has a valid end time
                if end_time is not None:
                    if latest_recorder is None or end_time > latest_recorder.info["end_time"]:
                        latest_recorder = recorder
                else:
                    print(f"Warning: Recorder {recorder_id} has no valid end time")
            except Exception as e:
                print(f"Error: {e}")

# Check if the latest recorder is found
if latest_recorder is None:
    print("No recorders found")
else:
    print(f"Latest recorder: {latest_recorder}")

    # Load the specified file from the latest recorder
    metrics = pd.Series(latest_recorder.list_metrics())

    # How much the model ranks. The backtest figures stand without these.
    try:
        rounds = best_iteration(latest_recorder.client.get_metric_history(latest_recorder.id, "l2.valid"))
        if rounds is not None:
            metrics["model.best_iteration"] = rounds
        topk = int(os.environ.get("RD_MARKET_TOPK") or 50)
        for name, value in ranking_stats(latest_recorder.load_object("pred.pkl"), topk).items():
            metrics[name] = value
    except Exception as e:
        print(f"Model health not measured: {e}")

    output_path = Path(__file__).resolve().parent / "qlib_res.csv"
    metrics.to_csv(output_path)

    print(f"Output has been saved to {output_path}")

    ret_data_frame = latest_recorder.load_object("portfolio_analysis/report_normal_1day.pkl")
ret_data_frame.to_parquet("ret.parquet")

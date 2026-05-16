import pandas as pd
import json
df = pd.read_csv('benchmark_paper_results.csv')
means = df[['RF_Acc', 'FLAML_Acc', 'Optuna_DL_Acc', 'RF_F1', 'FLAML_F1', 'Optuna_DL_F1', 'RF_Time', 'FLAML_Time', 'Optuna_DL_Time']].mean().to_dict()
with open('metrics.json', 'w') as f:
    json.dump(means, f, indent=4)

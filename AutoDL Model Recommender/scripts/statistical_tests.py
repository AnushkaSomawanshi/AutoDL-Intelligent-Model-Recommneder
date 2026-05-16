import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon
import numpy as np

def run_statistical_analysis(csv_file='benchmark_paper_results.csv', output_file='statistical_analysis_report.md'):
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"File {csv_file} not found.")
        return

    # Extract accuracy columns
    rf_acc = df['RF_Acc'].dropna()
    flaml_acc = df['FLAML_Acc'].dropna()
    dl_acc = df['Optuna_DL_Acc'].dropna()

    metrics_to_test = [('Accuracy', '_Acc'), ('F1-Score', '_F1'), ('Precision', '_Precision'), ('Recall', '_Recall')]
    report = ["# Comprehensive Statistical Significance Analysis\n"]
    report.append(f"**Number of benchmark datasets analyzed:** {len(df)}\n")

    for metric_name, suffix in metrics_to_test:
        rf_col = df[f'RF{suffix}'].dropna()
        flaml_col = df[f'FLAML{suffix}'].dropna()
        dl_col = df[f'Optuna_DL{suffix}'].dropna()

        valid_idx = rf_col.index.intersection(flaml_col.index).intersection(dl_col.index)
        rf_vals = rf_col.loc[valid_idx]
        flaml_vals = flaml_col.loc[valid_idx]
        dl_vals = dl_col.loc[valid_idx]

        report.append(f"## Metric: {metric_name}")
        report.append(f"**Valid Datasets:** {len(valid_idx)}")

        # Friedman
        stat, p_friedman = friedmanchisquare(rf_vals, flaml_vals, dl_vals)
        report.append(f"- **Friedman Test Statistic:** {stat:.4f}")
        report.append(f"- **Friedman P-Value:** {p_friedman:.4e}")
        
        if p_friedman < 0.05:
            report.append("✅ *Significant difference across all 3 models.*\n")
        else:
            report.append("❌ *No significant global difference across all 3 models at alpha=0.05.* (Post-hocs below are exploratory)\n")

        # Wilcoxon Pairwise
        report.append("### Pairwise Wilcoxon Signed-Rank Tests")
        pairs = [
            ("FLAML AutoML", flaml_vals, "Optuna DL", dl_vals),
            ("Random Forest", rf_vals, "Optuna DL", dl_vals),
            ("Random Forest", rf_vals, "FLAML AutoML", flaml_vals)
        ]
        
        report.append("| Comparison | P-value | Significant (α=0.05)? | Winner |")
        report.append("| --- | --- | --- | --- |")
        
        for name1, data1, name2, data2 in pairs:
            w, p = wilcoxon(data1, data2)
            sig = "Yes ✅" if p < 0.05 else "No ❌"
            winner = name1 if data1.mean() > data2.mean() else name2
            if p >= 0.05:
                winner = "Statistical Tie"
            report.append(f"| {name1} vs {name2} | {p:.4e} | {sig} | **{winner}** |")
        
        report.append("\n---\n")

    report_content = "\n".join(report)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report_content)

    print("Statistical Analysis generated successfully: " + output_file)

if __name__ == '__main__':
    run_statistical_analysis()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load the data
df = pd.read_csv('benchmark_paper_results.csv')

# Target artifact directory
# From user context: C:\Users\Palash Hemade\.gemini\antigravity\brain\1e55ad45-0a12-47d3-a735-51fba40a0372\artifacts
artifact_dir = r"C:\Users\Palash Hemade\.gemini\antigravity\brain\1e55ad45-0a12-47d3-a735-51fba40a0372\artifacts"
os.makedirs(artifact_dir, exist_ok=True)

# Set the style
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

# 1. Boxplot of Accuracy
plt.figure(figsize=(8, 6))
acc_data = pd.melt(df, value_vars=['RF_Acc', 'FLAML_Acc', 'Optuna_DL_Acc'], 
                   var_name='Model', value_name='Accuracy')
acc_data['Model'] = acc_data['Model'].map({'RF_Acc': 'Random Forest', 
                                           'FLAML_Acc': 'FLAML AutoML', 
                                           'Optuna_DL_Acc': 'Optuna DL'})
sns.boxplot(x='Model', y='Accuracy', data=acc_data, palette='Set2')
plt.title('Comparison of Model Robustness across 72 OpenML Datasets')
plt.ylabel('Test Accuracy')
plt.xlabel('Algorithm')
plt.tight_layout()
plt.savefig(os.path.join(artifact_dir, 'accuracy_boxplot.png'), dpi=300)
plt.close()

# 2. Time vs Accuracy Scatter
plt.figure(figsize=(8, 6))
plt.scatter(df['RF_Time'], df['RF_Acc'], alpha=0.6, label='Random Forest', color='teal')
plt.scatter(df['FLAML_Time'], df['FLAML_Acc'], alpha=0.6, label='FLAML AutoML', color='orange')
plt.scatter(df['Optuna_DL_Time'], df['Optuna_DL_Acc'], alpha=0.6, label='Optuna DL', color='purple')
plt.xscale('log')
plt.title('Training Time vs. Accuracy (Log Scale)')
plt.xlabel('Training Time (seconds)')
plt.ylabel('Test Accuracy')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(artifact_dir, 'time_vs_accuracy.png'), dpi=300)
plt.close()

# 3. Barplot of Average Metrics (F1, Precision, Recall)
plt.figure(figsize=(10, 6))
metrics = {
    'Metric': ['F1 Score', 'F1 Score', 'F1 Score',
               'Precision', 'Precision', 'Precision',
               'Recall', 'Recall', 'Recall'],
    'Model': ['Random Forest', 'FLAML', 'Optuna DL'] * 3,
    'Score': [
        df['RF_F1'].mean(), df['FLAML_F1'].mean(), df['Optuna_DL_F1'].mean(),
        df['RF_Precision'].mean(), df['FLAML_Precision'].mean(), df['Optuna_DL_Precision'].mean(),
        df['RF_Recall'].mean(), df['FLAML_Recall'].mean(), df['Optuna_DL_Recall'].mean()
    ]
}
metrics_df = pd.DataFrame(metrics)
sns.barplot(x='Metric', y='Score', hue='Model', data=metrics_df, palette='Set2')
plt.title('Average Classification Metrics Across All Datasets')
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig(os.path.join(artifact_dir, 'average_metrics.png'), dpi=300)
plt.close()

# Print out means for quick parsing
print("Means:")
print(df[['RF_Acc', 'FLAML_Acc', 'Optuna_DL_Acc']].mean())
print(df[['RF_Time', 'FLAML_Time', 'Optuna_DL_Time']].mean())

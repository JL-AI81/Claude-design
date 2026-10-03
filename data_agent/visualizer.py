import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend suitable for scripts
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


class Visualizer:
    """Generates charts and visualizations for datasets and research questions."""

    def __init__(self, output_dir: str = "output_reports/plots"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        sns.set_theme(style="whitegrid")

    def generate_plots_for_dataset(
        self, df: pd.DataFrame, dataset_name: str, questions: List[Dict[str, Any]]
    ) -> List[Dict[str, str]]:
        """
        Generates plots corresponding to questions and general dataset insights.
        Returns a list of dicts with plot metadata (title, filename, filepath).
        """
        generated_plots = []
        if df is None or df.empty:
            return generated_plots

        ds_plot_dir = self.output_dir / dataset_name
        ds_plot_dir.mkdir(parents=True, exist_ok=True)

        # 1. Correlation heatmap if 2 or more numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) >= 2:
            try:
                plt.figure(figsize=(8, 6))
                corr = df[numeric_cols].corr()
                sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", vmin=-1, vmax=1, square=True)
                plt.title(f"Korrelaatiomatriisi - {dataset_name}")
                plt.tight_layout()
                filename = f"{dataset_name}_correlation_matrix.png"
                filepath = ds_plot_dir / filename
                plt.savefig(filepath, dpi=150)
                plt.close()

                generated_plots.append({
                    "title": f"Korrelaatiomatriisi ({dataset_name})",
                    "filename": filename,
                    "filepath": str(filepath)
                })
            except Exception as e:
                plt.close()

        # 2. Plots for each research question
        for idx, q in enumerate(questions):
            plot_type = q.get("plot_type")
            col1 = q.get("col1")
            col2 = q.get("col2")

            if not col1 or col1 not in df.columns:
                continue

            try:
                plt.figure(figsize=(8, 5))
                filename = f"{dataset_name}_q{idx+1}_{plot_type}.png"
                filepath = ds_plot_dir / filename

                if plot_type == "scatter" and col2 and col2 in df.columns:
                    sns.scatterplot(data=df, x=col1, y=col2, color="steelblue", alpha=0.7)
                    sns.regplot(data=df, x=col1, y=col2, scatter=False, color="darkred")
                    plt.title(f"Hajontakuvio: {col1} vs {col2}")
                    plt.xlabel(col1)
                    plt.ylabel(col2)

                elif plot_type == "bar":
                    top_n = df[col1].value_counts().head(10)
                    y_vals = top_n.index.astype(str)
                    sns.barplot(x=top_n.values, y=y_vals, hue=y_vals, palette="viridis", legend=False)
                    plt.title(f"Luokkajakautuma: {col1}")
                    plt.xlabel("Lukumäärä")
                    plt.ylabel(col1)

                elif plot_type == "histogram":
                    sns.histplot(df[col1].dropna(), kde=True, color="teal")
                    plt.title(f"Histogrammi: {col1}")
                    plt.xlabel(col1)
                    plt.ylabel("Taajuus")

                elif plot_type == "boxplot" and col2 and col2 in df.columns:
                    top_cats = df[col1].value_counts().head(8).index
                    filtered_df = df[df[col1].isin(top_cats)]
                    sns.boxplot(data=filtered_df, x=col1, y=col2, hue=col1, palette="Set2", legend=False)
                    plt.title(f"Laatikkokuvio: {col2} ryhmittäin ({col1})")
                    plt.xlabel(col1)
                    plt.ylabel(col2)
                    plt.xticks(rotation=30)

                else:
                    plt.close()
                    continue

                plt.tight_layout()
                plt.savefig(filepath, dpi=150)
                plt.close()

                generated_plots.append({
                    "title": q["question"],
                    "filename": filename,
                    "filepath": str(filepath),
                    "question_id": q["id"]
                })

            except Exception as e:
                plt.close()

        return generated_plots

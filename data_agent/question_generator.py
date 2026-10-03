from typing import Dict, Any, List
import pandas as pd
import numpy as np


class QuestionGenerator:
    """Generates research questions and computes analytical answers based on dataset features."""

    def generate_questions(self, df: pd.DataFrame, dataset_name: str = "dataset") -> List[Dict[str, Any]]:
        """
        Analyzes the DataFrame and returns a list of research questions with computed answers,
        findings, and suggested plot types.
        """
        questions = []
        if df is None or df.empty:
            return questions

        numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
        categorical_cols = list(df.select_dtypes(include=['object', 'category', 'string']).columns)

        # Filter categorical columns to those with reasonable cardinality (e.g., <= 20 unique values)
        categorical_cols = [col for col in categorical_cols if 1 < df[col].nunique() <= 20]

        # 1. Correlation research questions (between top numeric pairs)
        if len(numeric_cols) >= 2:
            corr_matrix = df[numeric_cols].corr().abs()
            # Unstack and filter upper triangle
            pairs = []
            for i in range(len(numeric_cols)):
                for j in range(i + 1, len(numeric_cols)):
                    col1, col2 = numeric_cols[i], numeric_cols[j]
                    val = corr_matrix.loc[col1, col2]
                    if not np.isnan(val):
                        pairs.append((col1, col2, val))

            pairs.sort(key=lambda x: x[2], reverse=True)

            for col1, col2, corr_val in pairs[:2]:  # Top 2 correlations
                raw_corr = df[col1].corr(df[col2])
                direction = "positiivinen" if raw_corr > 0 else "negatiivinen"
                strength = "vahva" if abs(raw_corr) >= 0.7 else ("kohtalainen" if abs(raw_corr) >= 0.3 else "heikko")

                questions.append({
                    "id": f"q_corr_{col1}_{col2}",
                    "category": "Correlation",
                    "question": f"Miten sarakkeet '{col1}' ja '{col2}' korreloivat keskenään?",
                    "answer": f"Muuttujien välillä on {strength} {direction} korrelaatio (Pearson r = {raw_corr:.2f}).",
                    "plot_type": "scatter",
                    "col1": col1,
                    "col2": col2,
                    "value": raw_corr
                })

        # 2. Categorical distribution & top categories
        for cat_col in categorical_cols[:2]:
            counts = df[cat_col].value_counts()
            top_cat = counts.index[0]
            top_count = counts.iloc[0]
            top_pct = (top_count / len(df)) * 100

            questions.append({
                "id": f"q_cat_{cat_col}",
                "category": "Distribution",
                "question": f"Mikä on yleisin luokka sarakkeessa '{cat_col}' ja miten luokat jakautuvat?",
                "answer": f"Yleisin luokka on '{top_cat}', jonka osuus on {top_count} kpl ({top_pct:.1f}% kaikista riveistä). Yhteensä eri luokkia on {df[cat_col].nunique()} kpl.",
                "plot_type": "bar",
                "col1": cat_col,
                "col2": None
            })

        # 3. Numerical distribution & summary stats
        for num_col in numeric_cols[:2]:
            mean_val = df[num_col].mean()
            median_val = df[num_col].median()
            std_val = df[num_col].std()
            min_val = df[num_col].min()
            max_val = df[num_col].max()

            questions.append({
                "id": f"q_num_{num_col}",
                "category": "Summary",
                "question": f"Miten arvot jakautuvat sarakkeessa '{num_col}'?",
                "answer": f"Keskiarvo on {mean_val:.2f}, mediaani {median_val:.2f} ja keskihajonta {std_val:.2f}. Minimi on {min_val:.2f} ja maksimi {max_val:.2f}.",
                "plot_type": "histogram",
                "col1": num_col,
                "col2": None
            })

        # 4. Group Comparison (Categorical vs Numerical)
        if categorical_cols and numeric_cols:
            cat_col = categorical_cols[0]
            num_col = numeric_cols[0]

            grouped = df.groupby(cat_col)[num_col].mean().sort_values(ascending=False)
            if len(grouped) >= 2:
                highest_group = grouped.index[0]
                highest_val = grouped.iloc[0]
                lowest_group = grouped.index[-1]
                lowest_val = grouped.iloc[-1]

                questions.append({
                    "id": f"q_group_{cat_col}_{num_col}",
                    "category": "Group Comparison",
                    "question": f"Miten '{num_col}' eroaa eri ryhmien ('{cat_col}') välillä?",
                    "answer": f"Korkein keskiarvo muuttujalle '{num_col}' on ryhmässä '{highest_group}' ({highest_val:.2f}) ja matalin ryhmässä '{lowest_group}' ({lowest_val:.2f}).",
                    "plot_type": "boxplot",
                    "col1": cat_col,
                    "col2": num_col
                })

        return questions

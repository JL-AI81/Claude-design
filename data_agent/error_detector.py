from typing import Dict, Any, List
import pandas as pd
import numpy as np


class ErrorDetector:
    """Detects data quality issues and errors in pandas DataFrames."""

    def analyze(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Performs comprehensive error detection on a DataFrame.
        Returns a dictionary containing findings and summary statistics.
        """
        if df is None or df.empty:
            return {
                "total_rows": 0,
                "total_columns": 0,
                "is_empty": True,
                "issues": [{"type": "empty_dataset", "description": "Dataset on tyhjä tai sitä ei saatu ladattua."}],
                "missing_values": {},
                "duplicate_rows": 0,
                "outliers": {},
                "datatype_issues": [],
                "constant_columns": [],
                "all_null_columns": []
            }

        num_rows, num_cols = df.shape
        issues = []

        # 1. Duplicate rows
        duplicate_count = int(df.duplicated().sum())
        if duplicate_count > 0:
            pct = (duplicate_count / num_rows) * 100
            issues.append({
                "type": "duplicates",
                "count": duplicate_count,
                "percentage": round(pct, 2),
                "description": f"Löytyi {duplicate_count} täysin identtistä riviä ({pct:.1f}% data-joukosta)."
            })

        # 2. Missing values per column
        missing_values = {}
        for col in df.columns:
            null_count = int(df[col].isnull().sum())
            if null_count > 0:
                pct = (null_count / num_rows) * 100
                missing_values[col] = {
                    "count": null_count,
                    "percentage": round(pct, 2)
                }
                if pct > 50.0:
                    issues.append({
                        "type": "high_missing_values",
                        "column": col,
                        "count": null_count,
                        "percentage": round(pct, 2),
                        "description": f"Sarakkeessa '{col}' on erittäin paljon puuttuvia arvoja ({pct:.1f}%)."
                    })

        # 3. All null columns and Constant columns
        all_null_columns = [col for col, data in missing_values.items() if data["count"] == num_rows]
        constant_columns = []
        for col in df.columns:
            if col not in all_null_columns and df[col].nunique(dropna=True) == 1:
                constant_columns.append(col)
                issues.append({
                    "type": "constant_column",
                    "column": col,
                    "description": f"Sarakkeessa '{col}' on vain yksi uniikki arvo ({df[col].dropna().iloc[0]})."
                })

        # 4. Numerical outliers using IQR method
        outliers = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            col_data = df[col].dropna()
            if len(col_data) >= 5:
                q1 = col_data.quantile(0.25)
                q3 = col_data.quantile(0.75)
                iqr = q3 - q1
                if iqr > 0:
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    outlier_mask = (col_data < lower_bound) | (col_data > upper_bound)
                    outlier_count = int(outlier_mask.sum())
                    if outlier_count > 0:
                        pct = (outlier_count / len(col_data)) * 100
                        outliers[col] = {
                            "count": outlier_count,
                            "percentage": round(pct, 2),
                            "lower_bound": float(lower_bound),
                            "upper_bound": float(upper_bound)
                        }
                        if pct > 5.0:
                            issues.append({
                                "type": "outliers",
                                "column": col,
                                "count": outlier_count,
                                "percentage": round(pct, 2),
                                "description": f"Sarakkeessa '{col}' on {outlier_count} poikkeavaa arvoa (outlier, {pct:.1f}%)."
                            })

        # 5. Potential Datatype Issues (e.g. numeric stored as object, unparsed date strings)
        datatype_issues = []
        object_cols = df.select_dtypes(include=['object', 'string']).columns
        for col in object_cols:
            non_null_series = df[col].dropna().astype(str)
            if len(non_null_series) == 0:
                continue

            # Check if column could be numeric
            converted_numeric = pd.to_numeric(non_null_series.str.replace(',', '.'), errors='coerce')
            if converted_numeric.notnull().mean() > 0.8:
                datatype_issues.append({
                    "column": col,
                    "suggested_type": "numeric",
                    "description": f"Sarake '{col}' on merkkijonomuotoinen, mutta {converted_numeric.notnull().mean()*100:.0f}% arvoista on numeerisia."
                })
                issues.append({
                    "type": "type_mismatch",
                    "column": col,
                    "description": f"Sarake '{col}' pitäisi todennäköisesti muuntaa numeeriseksi."
                })
                continue

            # Check if column could be date/datetime
            if any(term in col.lower() for term in ['date', 'time', 'pvm', 'paiva', 'aika', 'created', 'updated']):
                converted_date = pd.to_datetime(non_null_series, errors='coerce')
                if converted_date.notnull().mean() > 0.7:
                    datatype_issues.append({
                        "column": col,
                        "suggested_type": "datetime",
                        "description": f"Sarake '{col}' vaikuttaa päivämäärältä, mutta on tallennettu tekstinä."
                    })
                    issues.append({
                        "type": "type_mismatch",
                        "column": col,
                        "description": f"Sarake '{col}' pitäisi muuntaa päivämäärämuotoon."
                    })

        return {
            "total_rows": num_rows,
            "total_columns": num_cols,
            "is_empty": False,
            "issues": issues,
            "missing_values": missing_values,
            "duplicate_rows": duplicate_count,
            "outliers": outliers,
            "datatype_issues": datatype_issues,
            "constant_columns": constant_columns,
            "all_null_columns": all_null_columns
        }

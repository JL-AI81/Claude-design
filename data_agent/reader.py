import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd


SUPPORTED_EXTENSIONS = {
    ".csv": "csv",
    ".xlsx": "excel",
    ".xls": "excel",
    ".json": "json",
    ".parquet": "parquet"
}


class DataReader:
    """Reads data files from a given directory or file path."""

    def __init__(self, target_path: str = r"F:\Data-allas"):
        self.target_path = Path(target_path)

    def discover_files(self, target_override: Optional[Path] = None) -> List[Path]:
        """Finds all supported data files in target path (file or directory)."""
        target = target_override if target_override else self.target_path
        files_found = []
        if not target.exists():
            return files_found

        if target.is_file():
            if target.suffix.lower() in SUPPORTED_EXTENSIONS:
                files_found.append(target)
        elif target.is_dir():
            for root, _, files in os.walk(target):
                for file in files:
                    file_p = Path(root) / file
                    if file_p.suffix.lower() in SUPPORTED_EXTENSIONS:
                        files_found.append(file_p)
        return files_found

    def load_dataset(self, file_path: Path) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """Loads a single data file into a pandas DataFrame."""
        try:
            ext = file_path.suffix.lower()
            if ext == ".csv":
                # Try common encodings and separators if default fails
                for sep in [",", ";", "\t", "|"]:
                    try:
                        df = pd.read_csv(file_path, sep=sep)
                        if df.shape[1] > 1 or sep == ",":
                            return df, None
                    except Exception:
                        continue
                df = pd.read_csv(file_path)
                return df, None
            elif ext in [".xlsx", ".xls"]:
                df = pd.read_excel(file_path)
                return df, None
            elif ext == ".json":
                df = pd.read_json(file_path)
                return df, None
            elif ext == ".parquet":
                df = pd.read_parquet(file_path)
                return df, None
            else:
                return None, f"Unsupported file extension: {ext}"
        except Exception as e:
            return None, str(e)

    def load_all(self, path_override: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        """
        Scans directory/file and loads all supported data files into DataFrames.
        Returns a dictionary mapping dataset_name -> {'df': DataFrame, 'path': Path, 'error': str}.
        """
        target = Path(path_override) if path_override else self.target_path
        results = {}

        files_to_process = self.discover_files(target_override=target)

        for file_p in files_to_process:
            dataset_name = file_p.stem
            df, error = self.load_dataset(file_p)
            results[dataset_name] = {
                "name": dataset_name,
                "path": str(file_p),
                "df": df,
                "error": error,
                "extension": file_p.suffix.lower()
            }

        return results

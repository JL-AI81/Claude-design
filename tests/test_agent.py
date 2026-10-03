import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import os

from data_agent.reader import DataReader
from data_agent.error_detector import ErrorDetector
from data_agent.question_generator import QuestionGenerator
from data_agent.visualizer import Visualizer
from data_agent.report import ReportGenerator
from data_agent.cli import run_agent


@pytest.fixture
def sample_df():
    data = {
        "id": [1, 2, 3, 4, 5, 5],  # duplicate row at end
        "category": ["A", "B", "A", "B", "C", "C"],
        "price": [10.0, 20.0, 15.0, 1000.0, 25.0, 25.0],  # outlier 1000.0
        "quantity": [5, 10, 15, 20, 25, 25],
        "constant_col": [100, 100, 100, 100, 100, 100]
    }
    return pd.DataFrame(data)


def test_error_detector(sample_df):
    detector = ErrorDetector()
    results = detector.analyze(sample_df)

    assert results["total_rows"] == 6
    assert results["total_columns"] == 5
    assert results["duplicate_rows"] == 1
    assert "constant_col" in results["constant_columns"]
    assert "price" in results["outliers"]
    assert len(results["issues"]) > 0


def test_question_generator(sample_df):
    q_gen = QuestionGenerator()
    questions = q_gen.generate_questions(sample_df, dataset_name="test_data")

    assert len(questions) > 0
    categories = [q["category"] for q in questions]
    assert any(c in ["Correlation", "Distribution", "Summary", "Group Comparison"] for c in categories)


def test_visualizer_and_report(sample_df):
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        csv_file = tmp_path / "test_dataset.csv"
        sample_df.to_csv(csv_file, index=False)

        # Test Reader
        reader = DataReader(target_path=str(tmp_path))
        loaded = reader.load_all()
        assert "test_dataset" in loaded
        df = loaded["test_dataset"]["df"]
        assert df is not None

        # Test Question Generator
        q_gen = QuestionGenerator()
        questions = q_gen.generate_questions(df, dataset_name="test_dataset")

        # Test Visualizer
        viz = Visualizer(output_dir=str(tmp_path / "plots"))
        plots = viz.generate_plots_for_dataset(df, "test_dataset", questions)
        assert len(plots) > 0
        for p in plots:
            assert os.path.exists(p["filepath"])

        # Test Report Generator
        quality = ErrorDetector().analyze(df)
        analysis_results = {
            "test_dataset": {
                "name": "test_dataset",
                "path": str(csv_file),
                "df": df,
                "error": None,
                "quality": quality,
                "questions": questions,
                "plots": plots
            }
        }
        reporter = ReportGenerator(output_dir=str(tmp_path / "reports"))
        report_path = reporter.generate_report(analysis_results)
        assert os.path.exists(report_path)
        with open(report_path, "r", encoding="utf-8") as f:
            content = f.read()
            assert "Data-altaan Automaattinen Analyysiraportti" in content
            assert "test_dataset" in content


def test_cli_execution():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        df = pd.DataFrame({"col_a": [1, 2, 3], "col_b": [4, 5, 6]})
        df.to_csv(data_dir / "sample.csv", index=False)

        out_dir = tmp_path / "output"
        run_agent(data_path=str(data_dir), output_dir=str(out_dir))

        report_file = out_dir / "Data_Analysis_Report.md"
        assert report_file.exists()

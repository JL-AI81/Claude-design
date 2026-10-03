import argparse
import sys
from pathlib import Path
from data_agent.reader import DataReader
from data_agent.error_detector import ErrorDetector
from data_agent.question_generator import QuestionGenerator
from data_agent.visualizer import Visualizer
from data_agent.report import ReportGenerator


def run_agent(data_path: str = r"F:\Data-allas", output_dir: str = "output_reports"):
    """
    Runs full Data Agent pipeline: reading data, detecting errors,
    generating research questions, creating visual plots, and saving report.
    """
    print("=" * 60)
    print("🤖 Data Agentti aloittaa analyysin...")
    print(f"📁 Kohdekansio/Tiedosto: {data_path}")
    print(f"📊 Raportointikansio: {output_dir}")
    print("=" * 60)

    reader = DataReader(target_path=data_path)
    detector = ErrorDetector()
    q_generator = QuestionGenerator()
    visualizer = Visualizer(output_dir=f"{output_dir}/plots")
    reporter = ReportGenerator(output_dir=output_dir)

    datasets = reader.load_all()

    if not datasets:
        print(f"⚠️ Ei löydetty tuettuja data-tiedostoja polusta: {data_path}")
        print("Huomio: Tarkista polku tai anna custom-polku argumenttina '--path'")
        return

    analysis_results = {}

    for name, info in datasets.items():
        print(f"\n🔍 Käsitellään aineistoa: {name}")
        df = info.get("df")
        err = info.get("error")

        if err or df is None:
            print(f"❌ Virhe ladattaessa tiedostoa {info['path']}: {err}")
            analysis_results[name] = {
                "name": name,
                "path": info["path"],
                "df": None,
                "error": err,
                "quality": {},
                "questions": [],
                "plots": []
            }
            continue

        # 1. Error Detection
        quality = detector.analyze(df)
        print(f"  • Rivejä: {quality['total_rows']}, Sarakkeita: {quality['total_columns']}")
        print(f"  • Havaitut laatuongelmat: {len(quality['issues'])} kpl")

        # 2. Question Generation
        questions = q_generator.generate_questions(df, dataset_name=name)
        print(f"  • Generoidut tutkimuskysymykset: {len(questions)} kpl")

        # 3. Visualization
        plots = visualizer.generate_plots_for_dataset(df, dataset_name=name, questions=questions)
        print(f"  • Luodut kuvaajat: {len(plots)} kpl")

        analysis_results[name] = {
            "name": name,
            "path": info["path"],
            "df": df,
            "error": None,
            "quality": quality,
            "questions": questions,
            "plots": plots
        }

    # 4. Generate Markdown Report
    report_file = reporter.generate_report(analysis_results)
    print("\n" + "=" * 60)
    print(f"✅ Analyysi valmis! Raportti tallennettu: {report_file}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Data Agentti - Automaattinen data-analyysi, virheiden havainnointi, tutkimuskysymykset ja kuvaajat."
    )
    parser.add_argument(
        "--path",
        type=str,
        default=r"F:\Data-allas",
        help="Polku kansion F:\\Data-allas dataan tai muuhun data-tiedostoon/kansioon."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output_reports",
        help="Kansio, johon analyysiraportti ja kuvaajat tallennetaan."
    )

    args = parser.parse_args()
    run_agent(data_path=args.path, output_dir=args.output)


if __name__ == "__main__":
    main()

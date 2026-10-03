from pathlib import Path
from typing import Dict, Any, List


class ReportGenerator:
    """Generates structured Markdown reports from data analysis, quality checks, questions, and plots."""

    def __init__(self, output_dir: str = "output_reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_report(self, analysis_results: Dict[str, Any]) -> str:
        """
        Compiles analysis results into a Markdown report.
        Saves the file to output_dir and returns the filepath.
        """
        report_path = self.output_dir / "Data_Analysis_Report.md"

        md_lines = [
            "# Data-altaan Automaattinen Analyysiraportti",
            "Generoinut: **Data Agentti**",
            "",
            "---",
            ""
        ]

        if not analysis_results:
            md_lines.append("Ei käsiteltäviä aineistoja tai kansion tiedostot olivat tyhjiä.\n")
            report_path.write_text("\n".join(md_lines), encoding="utf-8")
            return str(report_path)

        md_lines.append(f"## Yhteenveto kansion aineistoista ({len(analysis_results)} kpl)\n")

        for ds_name, ds_info in analysis_results.items():
            md_lines.append(f"### 📂 Aineisto: `{ds_name}`")
            md_lines.append(f"- **Tiedostopolku:** `{ds_info['path']}`")

            if ds_info.get("error"):
                md_lines.append(f"- ⚠️ **Virhe tiedostoa ladattaessa:** {ds_info['error']}\n")
                continue

            quality = ds_info.get("quality", {})
            md_lines.append(f"- **Rivejä:** {quality.get('total_rows', 0):,} kpl")
            md_lines.append(f"- **Sarakkeita:** {quality.get('total_columns', 0)} kpl\n")

            # 1. Havaitut virheet ja laatuongelmat
            md_lines.append("#### 🚨 Havaitut laatuongelmat ja poikkeamat")
            issues = quality.get("issues", [])
            if not issues:
                md_lines.append("- ✅ Ei merkittäviä laatuongelmia havaittu.\n")
            else:
                for issue in issues:
                    md_lines.append(f"- ⚠️ {issue.get('description')}")
                md_lines.append("")

            # 2. Tutkimuskysymykset ja tulokset
            md_lines.append("#### ❓ Automaattiset tutkimuskysymykset ja analyysi")
            questions = ds_info.get("questions", [])
            if not questions:
                md_lines.append("- Ei saatu muodostettua automaattisia kysymyksiä.\n")
            else:
                for idx, q in enumerate(questions, 1):
                    md_lines.append(f"**Kysymys {idx}: {q['question']}**")
                    md_lines.append(f"- **Tulos / Vastaus:** {q['answer']}\n")

            # 3. Kuvaajat
            md_lines.append("#### 📊 Luodut kuvaajat")
            plots = ds_info.get("plots", [])
            if not plots:
                md_lines.append("- Ei luotuja kuvaajia.\n")
            else:
                for p in plots:
                    # Relative path for Markdown preview
                    rel_filepath = p['filepath']
                    md_lines.append(f"![{p['title']}]({rel_filepath})")
                    md_lines.append(f"*{p['title']}*\n")

            md_lines.append("---\n")

        report_content = "\n".join(md_lines)
        report_path.write_text(report_content, encoding="utf-8")
        return str(report_path)

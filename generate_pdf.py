import argparse
import json
from fpdf import FPDF


def add_section_header(pdf, title):
    pdf.set_font("Arial", style="B", size=12)
    pdf.set_fill_color(220, 220, 220)
    pdf.cell(0, 10, txt=title, ln=True, fill=True)
    pdf.set_font("Arial", size=10)
    pdf.ln(2)


def add_metrics_block(pdf, label, mse, rmse, mae):
    pdf.set_font("Arial", style="B", size=10)
    pdf.cell(0, 8, txt=label, ln=True)
    pdf.set_font("Arial", size=10)
    pdf.cell(0, 7, txt=f"  MSE  : {mse:.6f}", ln=True)
    pdf.cell(0, 7, txt=f"  RMSE : {rmse:.6f}", ln=True)
    pdf.cell(0, 7, txt=f"  MAE  : {mae:.6f}", ln=True)
    pdf.ln(3)


def generate_pdf(metrics_file, output_file):
    with open(metrics_file) as f:
        metrics = json.load(f)

    report_type     = metrics.get("report_type", "report")
    model_type      = metrics.get("model_type", "N/A")
    search_strategy = metrics.get("search_strategy", "N/A")
    scoring         = metrics.get("scoring", "N/A")
    hyperparameters = metrics.get("hyperparameters", {})

    pdf = FPDF()
    pdf.add_page()

    # ── Title ──────────────────────────────────────────────────────────────────
    pdf.set_font("Arial", style="B", size=16)
    pdf.cell(0, 12, txt=f"Model {report_type.capitalize()} Report", ln=True, align="C")
    pdf.ln(4)

    # ── Search Configuration (training reports only) ───────────────────────────
    if report_type == "training":
        add_section_header(pdf, "Search Configuration")
        pdf.cell(0, 7, txt=f"  Model type      : {model_type}", ln=True)
        pdf.cell(0, 7, txt=f"  Search strategy : {search_strategy}", ln=True)
        pdf.cell(0, 7, txt=f"  Scoring metric  : {scoring}", ln=True)
        pdf.cell(0, 7, txt=f"  Iterations (n_iter) : {metrics.get('n_iter', 'N/A')}", ln=True)
        pdf.cell(0, 7, txt=f"  Cross-validation folds (cv) : {metrics.get('cv', 'N/A')}", ln=True)
        pdf.cell(0, 7, txt=f"  Best CV score   : {metrics.get('best_cv_score', 'N/A')}", ln=True)
        pdf.ln(4)

        # Best params found by search
        add_section_header(pdf, "Best Parameters Found")
        for key, value in metrics.get("best_params", {}).items():
            pdf.cell(0, 7, txt=f"  {key}: {value}", ln=True)
        pdf.ln(4)

    else:
        # For test reports, show model and strategy as header info
        add_section_header(pdf, "Model Configuration")
        pdf.cell(0, 7, txt=f"  Model type      : {model_type}", ln=True)
        pdf.cell(0, 7, txt=f"  Search strategy : {search_strategy}", ln=True)
        pdf.cell(0, 7, txt=f"  Scoring metric  : {scoring}", ln=True)
        pdf.ln(4)

    # ── Full Resolved Hyperparameters ──────────────────────────────────────────
    add_section_header(pdf, "Hyperparameters (Full Resolved Set)")
    for key, value in hyperparameters.items():
        pdf.cell(0, 7, txt=f"  {key}: {value}", ln=True)
    pdf.ln(4)

    # ── Error Metrics ──────────────────────────────────────────────────────────
    add_section_header(pdf, "Error Metrics")
    if report_type == "training":
        train = metrics.get("training", {})
        val   = metrics.get("validation", {})
        add_metrics_block(pdf, "Training Set",
                          train["mse"], train["rmse"], train["mae"])
        add_metrics_block(pdf, "Validation Set",
                          val["mse"], val["rmse"], val["mae"])
    else:
        test = metrics.get("test", {})
        add_metrics_block(pdf, "Test Set",
                          test["mse"], test["rmse"], test["mae"])

    # ── Classification Reports ─────────────────────────────────────────────────
    if report_type == "training":
        train = metrics.get("training", {})
        val   = metrics.get("validation", {})

        add_section_header(pdf, "Training Classification Report")
        pdf.set_font("Courier", size=9)
        pdf.multi_cell(0, 6, train.get("classification_report", "N/A"))
        pdf.ln(4)

        add_section_header(pdf, "Validation Classification Report")
        pdf.set_font("Courier", size=9)
        pdf.multi_cell(0, 6, val.get("classification_report", "N/A"))
        pdf.ln(4)

        # ── Search Results Summary (top 10) ────────────────────────────────────
        search_results = metrics.get("search_results", [])
        if search_results:
            pdf.add_page()
            add_section_header(pdf, "Search Results (Top 10 by Rank)")
            pdf.set_font("Courier", size=8)
            for entry in search_results[:10]:
                line = (
                    f"  Rank {entry['rank']:>3} | "
                    f"Score: {entry['mean_cv_score']:>10.6f} "
                    f"(±{entry['std_cv_score']:.6f}) | "
                    f"Params: {entry['params']}"
                )
                pdf.multi_cell(0, 6, line)
    else:
        test = metrics.get("test", {})
        add_section_header(pdf, "Test Classification Report")
        pdf.set_font("Courier", size=9)
        pdf.multi_cell(0, 6, test.get("classification_report", "N/A"))

    pdf.output(output_file)
    print(f"PDF report saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate PDF report for model performance.")
    parser.add_argument('--report', type=str, help="Path to save the PDF report.")
    parser.add_argument('--metrics', type=str, help="Path to the metrics JSON file.")

    args = parser.parse_args()
    generate_pdf(args.metrics, args.report)

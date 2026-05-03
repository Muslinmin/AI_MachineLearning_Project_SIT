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
    # Load metrics from JSON
    with open(metrics_file) as f:
        metrics = json.load(f)

    report_type = metrics.get("report_type", "report")
    hyperparameters = metrics.get("hyperparameters", {})

    pdf = FPDF()
    pdf.add_page()

    # ── Title ──────────────────────────────────────────────────────────────────
    pdf.set_font("Arial", style="B", size=16)
    pdf.cell(0, 12, txt=f"Model {report_type.capitalize()} Report", ln=True, align="C")
    pdf.ln(4)

    # ── Hyperparameters ────────────────────────────────────────────────────────
    add_section_header(pdf, "Hyperparameters")
    for key, value in hyperparameters.items():
        pdf.cell(0, 7, txt=f"  {key}: {value}", ln=True)
    pdf.ln(4)

    # ── Metrics ────────────────────────────────────────────────────────────────
    add_section_header(pdf, "Error Metrics")

    if report_type == "training":
        train = metrics.get("training", {})
        val = metrics.get("validation", {})
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
        val = metrics.get("validation", {})

        add_section_header(pdf, "Training Classification Report")
        pdf.set_font("Courier", size=9)
        pdf.multi_cell(0, 6, train.get("classification_report", "N/A"))
        pdf.ln(4)

        add_section_header(pdf, "Validation Classification Report")
        pdf.set_font("Courier", size=9)
        pdf.multi_cell(0, 6, val.get("classification_report", "N/A"))
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

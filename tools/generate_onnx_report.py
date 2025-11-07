"""Generate ONNX export error report and save ExportedProgram (.pt2).

Usage:
    python tools/generate_onnx_report.py --weights path/to/weights.pt --output-dir out

This script will:
 - load the model via `from ultralytics import YOLO` (lazy import)
 - prepare a dummy input (1,3,640,640) on CPU
 - call torch.onnx.export(..., report=True) and capture the returned report
 - if an ExportedProgram is returned, save it as `<output-dir>/exported_model.pt2`
 - write the full exception stack to `<output-dir>/export_error.txt` if export fails

Make sure you run this with the same Python + PyTorch environment that you used when hitting the error.
"""
import argparse
import io
import os
import traceback

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, help="Path to model weights or model spec accepted by ultralytics YOLO")
    parser.add_argument("--output-dir", default="./onnx_report", help="Directory to write report, pt2, and error logs")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version to use")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    err_path = os.path.join(args.output_dir, "export_error.txt")
    pt2_path = os.path.join(args.output_dir, "exported_model.pt2")
    report_path = os.path.join(args.output_dir, "export_report.txt")

    # Load model using ultralytics API (adjust if you use a different loader)
    try:
        from ultralytics import YOLO
    except Exception as e:
        with open(err_path, "w") as f:
            f.write("Failed to import ultralytics.YOLO:\n")
            traceback.print_exc(file=f)
        raise

    model = YOLO(args.weights)
    model.eval()

    # Prepare dummy input on CPU (adjust shape to match your model if needed)
    dummy_input = torch.randn(1, 3, 640, 640)

    # Capture stdout/stderr for the export report
    buf = io.StringIO()

    try:
        # torch.onnx.export may return an ExportedProgram or None depending on torch version.
        # We pass report=True to request a structured report when available.
        exported = torch.onnx.export(
            model,
            dummy_input,
            os.path.join(args.output_dir, "model.onnx"),
            opset_version=args.opset,
            do_constant_folding=True,
            verbose=False,
            training=torch.onnx.TrainingMode.EVAL,
            report=True,
        )

        # If an ExportedProgram or similar is returned, try to save it
        if exported is not None:
            try:
                torch.save(exported, pt2_path)
                print(f"Saved ExportedProgram to: {pt2_path}")
            except Exception:
                # Some torch return types may not be directly saveable; write repr()
                with open(report_path + ".repr.txt", "w") as f:
                    f.write(repr(exported))

    except Exception:
        # Write full exception stack to file for filing an issue
        with open(err_path, "w") as f:
            f.write("Exception during torch.onnx.export:\n")
            traceback.print_exc(file=f)
        print(f"Export failed; full traceback written to: {err_path}")
        raise


if __name__ == "__main__":
    main()

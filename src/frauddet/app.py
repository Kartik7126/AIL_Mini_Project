from __future__ import annotations

import json
from io import BytesIO, StringIO
from pathlib import Path

import pandas as pd
from flask import Flask, flash, redirect, render_template, request, send_from_directory, url_for

from frauddet.config import BEST_MODEL_PATH, METRICS_PATH, REPORTS_DIR, ROOT, TARGET

_TEMPLATES = Path(__file__).resolve().parent / "templates"
from frauddet.predict import load_bundle, score_frame

SAMPLE_CSV = ROOT / "examples" / "sample_transactions.csv"
RESULT_COLUMNS = ["Time", "Amount", "anomaly_score", "predicted_fraud"]


def create_app() -> Flask:
    app = Flask(
        __name__,
        template_folder=str(_TEMPLATES),
    )
    app.secret_key = "frauddet-local-ui"

    def _metrics() -> dict | None:
        if not METRICS_PATH.exists():
            return None
        return json.loads(METRICS_PATH.read_text())

    def _model_ready() -> bool:
        return BEST_MODEL_PATH.exists()

    def _table(df: pd.DataFrame) -> list[dict]:
        cols = [c for c in RESULT_COLUMNS if c in df.columns]
        if TARGET in df.columns and TARGET not in cols:
            cols.append(TARGET)
        view = df[cols].copy()
        view["anomaly_score"] = view["anomaly_score"].map(lambda x: round(float(x), 4))
        if "Amount" in view.columns:
            view["Amount"] = view["Amount"].map(lambda x: round(float(x), 2))
        return view.to_dict(orient="records")

    @app.get("/")
    def index():
        metrics = _metrics()
        best = metrics.get("best_anomaly_model") if metrics else None
        model_stats = None
        if metrics and best:
            model_stats = metrics["models"].get(best)
        return render_template(
            "index.html",
            model_ready=_model_ready(),
            metrics=metrics,
            best=best,
            model_stats=model_stats,
            plots=_plot_files(),
            rows=None,
            n_flagged=None,
        )

    @app.get("/reports/<path:filename>")
    def reports(filename: str):
        return send_from_directory(REPORTS_DIR, filename)

    @app.post("/score-sample")
    def score_sample():
        if not _model_ready():
            flash("Train a model first: python -m frauddet train")
            return redirect(url_for("index"))
        if not SAMPLE_CSV.exists():
            flash("Missing examples/sample_transactions.csv")
            return redirect(url_for("index"))
        return _render_scored(pd.read_csv(SAMPLE_CSV), source="sample file")

    @app.post("/score-upload")
    def score_upload():
        if not _model_ready():
            flash("Train a model first: python -m frauddet train")
            return redirect(url_for("index"))
        uploaded = request.files.get("file")
        if uploaded is None or uploaded.filename == "":
            flash("Choose a CSV file first.")
            return redirect(url_for("index"))
        raw = uploaded.read()
        try:
            df = pd.read_csv(BytesIO(raw))
        except Exception:
            df = pd.read_csv(StringIO(raw.decode("utf-8", errors="replace")))
        return _render_scored(df, source=uploaded.filename)

    def _render_scored(df: pd.DataFrame, source: str):
        try:
            scored = score_frame(df, load_bundle())
        except Exception as exc:  # noqa: BLE001
            flash(f"Could not score that file: {exc}")
            return redirect(url_for("index"))
        metrics = _metrics()
        best = metrics.get("best_anomaly_model") if metrics else None
        model_stats = metrics["models"].get(best) if metrics and best else None
        n_flagged = int(scored["predicted_fraud"].sum())
        flash(f"Scored {len(scored)} rows from {source}. Flagged {n_flagged} as fraud.")
        return render_template(
            "index.html",
            model_ready=True,
            metrics=metrics,
            best=best,
            model_stats=model_stats,
            plots=_plot_files(),
            rows=_table(scored),
            n_flagged=n_flagged,
        )

    return app


def _plot_files() -> list[str]:
    names = ["class_balance.png", "amount_by_class.png", "pr_curves.png", "roc_curves.png"]
    return [n for n in names if (REPORTS_DIR / n).exists()]


def run_ui(host: str = "127.0.0.1", port: int = 7860) -> None:
    app = create_app()
    print(f"Open http://{host}:{port}")
    app.run(host=host, port=port, debug=False)

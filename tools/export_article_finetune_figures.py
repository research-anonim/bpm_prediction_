"""Export frozen versus fine-tuned article figures from run metric bundles."""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

try:
    from tools import export_article_complexity_figures as complexity
except ModuleNotFoundError:  # Direct ``python tools/<script>.py`` execution.
    import export_article_complexity_figures as complexity  # type: ignore[no-redef]


MODEL_ORDER = complexity.MODEL_ORDER
COMPLEXITIES = complexity.COMPLEXITIES
CDLG_PATTERN = re.compile(r"_CDLG-(simple|medium|complex)(\d+)_", re.IGNORECASE)
FIGURES = {
    f"Fig{number}": replace(spec, name=f"Fig{number}")
    for number, spec in zip(range(11, 15), complexity.FIGURES.values())
}


def _load_finetune_index(input_dir: Path) -> tuple[set[str], set[tuple[str, str]]]:
    path = input_dir / "drift" / "finetune_update_index.csv"
    if not path.exists():
        raise FileNotFoundError(f"Finetune index CSV not found: {path}")
    models: set[str] = set()
    run_ids: set[tuple[str, str]] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = {"paper_model", "run_id"} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} missing required columns: {sorted(missing)}")
        for row in reader:
            model = row["paper_model"].strip()
            run_id = row["run_id"].strip()
            if model and run_id:
                models.add(model)
                run_ids.add((model, run_id))
    if not models:
        raise ValueError(f"{path} has no eligible paper_model with a run_id")
    unsupported = models - set(MODEL_ORDER)
    if unsupported:
        raise ValueError(f"{path} contains unsupported paper_model: {sorted(unsupported)}")
    return models, run_ids


def _read_metric_rows(
    path: Path,
    max_step: int | None,
    eligible_models: set[str],
    finetune_run_ids: set[tuple[str, str]],
) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Metric CSV not found: {path}")
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = {"paper_model", "run_id", "step", "value", "dataset_complexity"} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} missing required columns: {sorted(missing)}")
        for row in reader:
            model = row["paper_model"].strip()
            if model not in eligible_models:
                continue
            complexity_name = row["dataset_complexity"].strip().lower()
            run_name = row.get("preset_name", "") or row.get("run_name", "")
            match = CDLG_PATTERN.search(run_name)
            if not complexity_name and match:
                complexity_name = match.group(1).lower()
            if complexity_name == "medium":
                complexity_name = "middle"
            if not complexity_name:
                complexity_name = "all"
            if complexity_name not in (*COMPLEXITIES, "all"):
                raise ValueError(f"Unknown dataset complexity {complexity_name!r} in {path} for run {row['run_id']}")
            try:
                step = int(float(row["step"]))
                float(row["value"])
            except (TypeError, ValueError):
                continue
            if max_step is not None and step > max_step:
                continue
            is_finetune = (model, row["run_id"].strip()) in finetune_run_ids or "-finetune" in run_name.lower()
            mode = "finetune" if is_finetune else "frozen"
            normalized = dict(row)
            normalized["paper_model"] = f"{model} {mode}"
            normalized["dataset_complexity"] = complexity_name
            normalized["step"] = str(step)
            if match:
                normalized["dataset_id"] = f"{match.group(1).lower()}{match.group(2)}"
            rows.append(normalized)
    return rows


def _hierarchical_aggregate(rows: list[dict[str, str]]):
    return complexity._hierarchical_aggregate(rows)


def _aggregate_loan(rows: list[dict[str, str]]):
    by_seed = defaultdict(list)
    for row in rows:
        seed = row.get("seed", "").strip() or row["run_id"].strip()
        by_seed[(row["paper_model"], seed, int(row["step"]))].append(float(row["value"]))
    by_step = defaultdict(list)
    for (model, _seed, step), values in by_seed.items():
        by_step[(model, step)].append(sum(values) / len(values))
    series = {}
    for (model, step), values in sorted(by_step.items()):
        series.setdefault(model, []).append((step, sum(values) / len(values), min(values), max(values)))
    return {"all": series}


def _plot_panel(ax, series, panel, complexity_name, xlabel, line_width_scale, model_order):
    for model in model_order:
        for mode, linestyle in (("frozen", "-"), ("finetune", "--")):
            label = f"{model} {mode}"
            points = series.get(label)
            if not points:
                continue
            xs = [point[0] for point in points]
            means = [point[1] for point in points]
            lows = [point[2] for point in points]
            highs = [point[3] for point in points]
            color = complexity.MODEL_COLORS[model]
            ax.fill_between(xs, lows, highs, color=color, alpha=complexity.MODEL_BAND_ALPHA[model], linewidth=0)
            ax.plot(xs, means, label=label, color=color, linestyle=linestyle, linewidth=(1.75 if model == "EOPKG" else 1.45) * line_width_scale)

    title = panel.title if complexity_name == "all" else f"{complexity_name.title()} - {panel.title}"
    ax.set_title(title, fontsize=11, pad=16)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(panel.ylabel, fontsize=10)
    bounds = complexity._panel_y_bounds(panel.metric)
    if bounds is None:
        values = [bound for points in series.values() for _, _, low, high in points for bound in (low, high)]
        if values:
            lower, upper = min(values), max(values)
            padding = max((upper - lower) * 0.05, 0.05)
            ax.set_ylim(lower - padding, upper + padding)
    else:
        ax.set_ylim(*bounds)
    ax.grid(True, color="#D9D9D9", linewidth=0.7, alpha=0.75)
    ax.tick_params(axis="both", labelsize=9)


def _render_figure(input_dir, output_dir, spec, formats, dpi, eligible_models, finetune_run_ids):
    model_order = [model for model in MODEL_ORDER if model in eligible_models]
    panel_rows = [
        _read_metric_rows(input_dir / spec.source / f"{panel.metric}.csv", spec.max_step, eligible_models, finetune_run_ids)
        for panel in spec.panels
    ]
    groups = {row["dataset_complexity"] for rows in panel_rows for row in rows}
    if not groups:
        raise ValueError(f"No eligible metric rows for {spec.name} in {input_dir}")
    if "all" in groups and len(groups) > 1:
        raise ValueError(f"Mixed grouped and ungrouped metric rows for {spec.name} in {input_dir}")
    panel_names = ("all",) if groups == {"all"} else COMPLEXITIES
    panel_series = [
        _aggregate_loan(rows) if panel_names == ("all",) else _hierarchical_aggregate(rows)
        for rows in panel_rows
    ]
    row_count, col_count = len(spec.panels), len(panel_names)
    size = (7.3, 4.8 if row_count == 1 else 3.4 * row_count) if panel_names == ("all",) else complexity._figure_size(spec)
    fig, axes = plt.subplots(row_count, col_count, figsize=size, squeeze=False)
    drift_steps = [
        point[0]
        for item in panel_series
        for complexity_series in item.values()
        for points in complexity_series.values()
        for point in points
    ]
    drift_bounds = (min(drift_steps), max(drift_steps)) if drift_steps else None
    for panel_index, panel in enumerate(spec.panels):
        for complexity_index, complexity_name in enumerate(panel_names):
            ax = axes[panel_index][complexity_index]
            _plot_panel(ax, panel_series[panel_index].get(complexity_name, {}), panel, complexity_name, spec.xlabel, spec.line_width_scale, model_order)
            if spec.source == "drift" and drift_bounds is not None:
                complexity.base._draw_version_boundaries(ax, *drift_bounds)

    label_to_handle = {}
    for row in axes:
        for ax in row:
            handles, labels = ax.get_legend_handles_labels()
            label_to_handle.update(zip(labels, handles))
    ordered_labels = [f"{model} {mode}" for model in model_order for mode in ("frozen", "finetune") if f"{model} {mode}" in label_to_handle]
    if ordered_labels:
        fig.legend([label_to_handle[label] for label in ordered_labels], ordered_labels, loc="lower center", ncol=len(ordered_labels), frameon=False, fontsize=10, bbox_to_anchor=(0.5, 0.01))
    fig.tight_layout(rect=(0.0, 0.06, 1.0, 0.92) if spec.source == "drift" else (0.0, 0.06, 1.0, 1.0))

    saved = []
    for fmt in formats:
        target_dir = output_dir / fmt
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / f"{spec.name}.{fmt}"
        kwargs = {"bbox_inches": "tight"}
        if fmt == "png":
            kwargs["dpi"] = dpi
        fig.savefig(target, format=fmt, **kwargs)
        saved.append(target)
    plt.close(fig)
    return saved


def _select_figures(names):
    requested = [item.strip() for item in names.split(",") if item.strip()]
    if not requested or requested == ["all"]:
        return list(FIGURES.values())
    unknown = [name for name in requested if name not in FIGURES]
    if unknown:
        raise argparse.ArgumentTypeError(f"Unknown figure(s): {', '.join(unknown)}")
    return [FIGURES[name] for name in requested]


def parse_args():
    parser = argparse.ArgumentParser(description="Export frozen versus finetune article figures.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--figures", default="all", help="Comma-separated figure ids, e.g. Fig11,Fig12.")
    parser.add_argument("--formats", type=complexity._parse_formats, default=["svg", "png"])
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def main():
    args = parse_args()
    eligible_models, finetune_run_ids = _load_finetune_index(args.input_dir)
    saved = []
    for spec in _select_figures(args.figures):
        saved.extend(_render_figure(args.input_dir, args.output_dir, spec, args.formats, args.dpi, eligible_models, finetune_run_ids))
    print("Exported article finetune figures:")
    for path in saved:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

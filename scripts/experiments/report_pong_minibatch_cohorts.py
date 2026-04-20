#!/usr/bin/env python3
"""Summarize multiple fixed-minibatch matched-replay cohorts in one report."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

import report_pong_matched_replay as matched


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cohort-manifest",
        action="append",
        required=True,
        help="Path to a cohort manifest produced by run_pong_matched_minibatch_cohort.py. Pass multiple times.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for the combined report bundle.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cohorts = [_load_cohort(Path(path).resolve()) for path in args.cohort_manifest]
    cohorts.sort(key=lambda cohort: cohort["minibatch_size"])

    combined_rows: list[dict[str, float | str]] = []
    summary: dict[str, object] = {"cohorts": []}
    for cohort in cohorts:
        rows = cohort["rows"]
        delta_episode = [float(row["episode_return_delta"]) for row in rows]
        delta_perf = [float(row["perf_delta"]) for row in rows]
        delta_sps = [float(row["sps_delta"]) for row in rows]
        entry = {
            "minibatch_size": cohort["minibatch_size"],
            "num_pairs": len(rows),
            "native_final_episode_return_mean": statistics.fmean(float(row["native_final_episode_return"]) for row in rows),
            "generated_final_episode_return_mean": statistics.fmean(float(row["generated_final_episode_return"]) for row in rows),
            "native_final_perf_mean": statistics.fmean(float(row["native_final_perf"]) for row in rows),
            "generated_final_perf_mean": statistics.fmean(float(row["generated_final_perf"]) for row in rows),
            "native_final_sps_mean": statistics.fmean(float(row["native_final_sps"]) for row in rows),
            "generated_final_sps_mean": statistics.fmean(float(row["generated_final_sps"]) for row in rows),
            "delta": {
                "episode_return": matched._delta_summary(delta_episode),
                "perf": matched._delta_summary(delta_perf),
                "sps": matched._delta_summary(delta_sps),
            },
            "report_dir": str(cohort["report_dir"]),
            "cohort_manifest": str(cohort["manifest_path"]),
        }
        summary["cohorts"].append(entry)
        for row in rows:
            combined_row = dict(row)
            combined_row["cohort_label"] = f"mb={cohort['minibatch_size']}"
            combined_row["minibatch_size"] = float(cohort["minibatch_size"])
            combined_rows.append(combined_row)

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "combined_pairs.csv").write_text(_combined_csv(combined_rows), encoding="utf-8")
    (output_dir / "sps_delta_by_minibatch.svg").write_text(
        _grouped_delta_svg(
            cohorts,
            metric_key="sps_delta",
            title="Generated - Native SPS Delta by Minibatch Cohort",
            subtitle="5 matched native-picked runs per cohort",
        ),
        encoding="utf-8",
    )
    (output_dir / "perf_delta_by_minibatch.svg").write_text(
        _grouped_delta_svg(
            cohorts,
            metric_key="perf_delta",
            title="Generated - Native Perf Delta by Minibatch Cohort",
            subtitle="5 matched native-picked runs per cohort",
        ),
        encoding="utf-8",
    )
    (output_dir / "episode_return_delta_by_minibatch.svg").write_text(
        _grouped_delta_svg(
            cohorts,
            metric_key="episode_return_delta",
            title="Generated - Native Episode Return Delta by Minibatch Cohort",
            subtitle="5 matched native-picked runs per cohort",
        ),
        encoding="utf-8",
    )
    (output_dir / "final_sps_native_vs_generated.svg").write_text(
        _native_generated_group_svg(
            cohorts,
            native_key="native_final_sps",
            generated_key="generated_final_sps",
            title="Final SPS by Minibatch Cohort",
            subtitle="Matched native vs generated runs",
            y_label="final SPS",
        ),
        encoding="utf-8",
    )
    (output_dir / "final_perf_native_vs_generated.svg").write_text(
        _native_generated_group_svg(
            cohorts,
            native_key="native_final_perf",
            generated_key="generated_final_perf",
            title="Final Perf by Minibatch Cohort",
            subtitle="Matched native vs generated runs",
            y_label="final perf",
        ),
        encoding="utf-8",
    )
    (output_dir / "report.md").write_text(_report_markdown(cohorts, output_dir), encoding="utf-8")
    print(output_dir)
    return 0


def _load_cohort(manifest_path: Path) -> dict[str, object]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    report_dir = Path(manifest["report_dir"]).resolve()
    rows = []
    with (report_dir / "paired_runs.csv").open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            parsed = {key: _parse_cell(value) for key, value in row.items()}
            rows.append(parsed)
    return {
        "manifest_path": manifest_path,
        "minibatch_size": int(manifest["minibatch_size"]),
        "report_dir": report_dir,
        "rows": rows,
    }


def _parse_cell(value: str) -> float | str:
    try:
        return float(value)
    except ValueError:
        return value


def _combined_csv(rows: list[dict[str, float | str]]) -> str:
    headers = [
        "cohort_label",
        "minibatch_size",
        "native_run_id",
        "native_final_episode_return",
        "generated_final_episode_return",
        "episode_return_delta",
        "native_final_perf",
        "generated_final_perf",
        "perf_delta",
        "native_final_sps",
        "generated_final_sps",
        "sps_delta",
        "hidden_size",
        "num_layers",
        "model_size_proxy",
        "batch_size",
        "replay_ratio",
        "learning_rate",
    ]
    lines = [",".join(headers)]
    for row in rows:
        lines.append(",".join(str(row[key]) for key in headers))
    return "\n".join(lines) + "\n"


def _grouped_delta_svg(
    cohorts: list[dict[str, object]],
    *,
    metric_key: str,
    title: str,
    subtitle: str,
) -> str:
    width = 980
    height = 560
    pad_left = 90
    pad_right = 30
    pad_top = 70
    pad_bottom = 90
    plot_width = width - pad_left - pad_right
    plot_height = height - pad_top - pad_bottom

    all_values = [float(row[metric_key]) for cohort in cohorts for row in cohort["rows"]]
    y_min = min(all_values)
    y_max = max(all_values)
    limit = max(abs(y_min), abs(y_max))
    if limit == 0:
        limit = 1.0
    y_min = -limit
    y_max = limit

    def y_px(value: float) -> float:
        return pad_top + (y_max - value) / (y_max - y_min) * plot_height

    centers = [pad_left + plot_width * (index + 0.5) / len(cohorts) for index in range(len(cohorts))]
    zero_y = y_px(0.0)
    y_ticks = [y_min + (y_max - y_min) * i / 6 for i in range(7)]

    shapes: list[str] = []
    labels: list[str] = []
    for center, cohort in zip(centers, cohorts, strict=True):
        rows = cohort["rows"]
        values = [float(row[metric_key]) for row in rows]
        mean_value = statistics.fmean(values)
        median_value = statistics.median(values)
        cohort_width = min(240, plot_width / len(cohorts) * 0.6)
        step = cohort_width / max(1, len(values) - 1)
        xs = [center - cohort_width / 2 + index * step for index in range(len(values))]
        for x, row, value in zip(xs, rows, values, strict=True):
            color = "#16a34a" if value >= 0 else "#dc2626"
            shapes.append(
                f'<circle cx="{x:.2f}" cy="{y_px(value):.2f}" r="7" fill="{color}" opacity="0.85">'
                f'<title>run {row["native_run_id"]}: {metric_key}={value:.6g}</title></circle>'
            )
        mean_y = y_px(mean_value)
        median_y = y_px(median_value)
        shapes.append(
            f'<line x1="{center - cohort_width / 2:.2f}" x2="{center + cohort_width / 2:.2f}" '
            f'y1="{mean_y:.2f}" y2="{mean_y:.2f}" stroke="#1d4ed8" stroke-width="3" />'
        )
        shapes.append(
            f'<line x1="{center - cohort_width / 2:.2f}" x2="{center + cohort_width / 2:.2f}" '
            f'y1="{median_y:.2f}" y2="{median_y:.2f}" stroke="#0f172a" stroke-width="2" stroke-dasharray="6 4" />'
        )
        labels.append(
            f'<text x="{center:.2f}" y="{pad_top + plot_height + 28}" text-anchor="middle" '
            f'font-family="sans-serif" font-size="13" fill="#0f172a">mb={cohort["minibatch_size"]}</text>'
        )
        labels.append(
            f'<text x="{center:.2f}" y="{pad_top + plot_height + 46}" text-anchor="middle" '
            f'font-family="sans-serif" font-size="12" fill="#475569">mean {mean_value:.3g}</text>'
        )

    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="{width}" height="{height}" fill="#f8fafc" />
<text x="{width/2:.0f}" y="28" text-anchor="middle" font-family="sans-serif" font-size="20" fill="#0f172a">{title}</text>
<text x="{width/2:.0f}" y="48" text-anchor="middle" font-family="sans-serif" font-size="13" fill="#475569">{subtitle}</text>
<text x="26" y="{height/2:.0f}" transform="rotate(-90 26 {height/2:.0f})" text-anchor="middle" font-family="sans-serif" font-size="14" fill="#334155">{metric_key}</text>
<rect x="{pad_left}" y="{pad_top}" width="{plot_width}" height="{plot_height}" fill="#ffffff" stroke="#cbd5e1" stroke-width="1" />
{''.join(f'<line x1="{pad_left}" x2="{pad_left + plot_width}" y1="{y_px(tick):.2f}" y2="{y_px(tick):.2f}" stroke="#e2e8f0" stroke-width="1" />' for tick in y_ticks)}
<line x1="{pad_left}" x2="{pad_left + plot_width}" y1="{zero_y:.2f}" y2="{zero_y:.2f}" stroke="#0f172a" stroke-width="1.5" />
{''.join(shapes)}
{''.join(f'<text x="{pad_left - 10}" y="{y_px(tick) + 4:.2f}" text-anchor="end" font-family="sans-serif" font-size="12" fill="#475569">{tick:.3g}</text>' for tick in y_ticks)}
{''.join(labels)}
<g transform="translate({pad_left + 20}, {pad_top + 18})">
  <circle cx="7" cy="7" r="6" fill="#16a34a" opacity="0.85" />
  <text x="20" y="11" font-family="sans-serif" font-size="12" fill="#0f172a">positive delta</text>
  <circle cx="125" cy="7" r="6" fill="#dc2626" opacity="0.85" />
  <text x="138" y="11" font-family="sans-serif" font-size="12" fill="#0f172a">negative delta</text>
  <line x1="260" x2="290" y1="7" y2="7" stroke="#1d4ed8" stroke-width="3" />
  <text x="298" y="11" font-family="sans-serif" font-size="12" fill="#0f172a">mean</text>
  <line x1="350" x2="380" y1="7" y2="7" stroke="#0f172a" stroke-width="2" stroke-dasharray="6 4" />
  <text x="388" y="11" font-family="sans-serif" font-size="12" fill="#0f172a">median</text>
</g>
</svg>
"""


def _native_generated_group_svg(
    cohorts: list[dict[str, object]],
    *,
    native_key: str,
    generated_key: str,
    title: str,
    subtitle: str,
    y_label: str,
) -> str:
    width = 980
    height = 560
    pad_left = 90
    pad_right = 30
    pad_top = 70
    pad_bottom = 90
    plot_width = width - pad_left - pad_right
    plot_height = height - pad_top - pad_bottom

    all_values = [
        float(row[native_key])
        for cohort in cohorts
        for row in cohort["rows"]
    ] + [
        float(row[generated_key])
        for cohort in cohorts
        for row in cohort["rows"]
    ]
    y_min = min(all_values)
    y_max = max(all_values)
    if y_min == y_max:
        y_min -= 1.0
        y_max += 1.0

    def y_px(value: float) -> float:
        return pad_top + (y_max - value) / (y_max - y_min) * plot_height

    centers = [pad_left + plot_width * (index + 0.5) / len(cohorts) for index in range(len(cohorts))]
    y_ticks = [y_min + (y_max - y_min) * i / 6 for i in range(7)]
    shapes: list[str] = []
    labels: list[str] = []
    for center, cohort in zip(centers, cohorts, strict=True):
        rows = cohort["rows"]
        cohort_width = min(240, plot_width / len(cohorts) * 0.6)
        native_center = center - cohort_width * 0.18
        generated_center = center + cohort_width * 0.18
        native_values = [float(row[native_key]) for row in rows]
        generated_values = [float(row[generated_key]) for row in rows]
        native_step = cohort_width * 0.28 / max(1, len(rows) - 1)
        generated_step = cohort_width * 0.28 / max(1, len(rows) - 1)
        native_xs = [native_center - cohort_width * 0.14 + index * native_step for index in range(len(rows))]
        generated_xs = [generated_center - cohort_width * 0.14 + index * generated_step for index in range(len(rows))]

        for x, row, value in zip(native_xs, rows, native_values, strict=True):
            shapes.append(
                f'<circle cx="{x:.2f}" cy="{y_px(value):.2f}" r="6" fill="#2563eb" opacity="0.85">'
                f'<title>native run {row["native_run_id"]}: {native_key}={value:.6g}</title></circle>'
            )
        for x, row, value in zip(generated_xs, rows, generated_values, strict=True):
            shapes.append(
                f'<circle cx="{x:.2f}" cy="{y_px(value):.2f}" r="6" fill="#16a34a" opacity="0.85">'
                f'<title>generated run {row["native_run_id"]}: {generated_key}={value:.6g}</title></circle>'
            )
        native_mean = statistics.fmean(native_values)
        generated_mean = statistics.fmean(generated_values)
        shapes.append(
            f'<line x1="{native_center - cohort_width * 0.18:.2f}" x2="{native_center + cohort_width * 0.18:.2f}" '
            f'y1="{y_px(native_mean):.2f}" y2="{y_px(native_mean):.2f}" stroke="#1d4ed8" stroke-width="3" />'
        )
        shapes.append(
            f'<line x1="{generated_center - cohort_width * 0.18:.2f}" x2="{generated_center + cohort_width * 0.18:.2f}" '
            f'y1="{y_px(generated_mean):.2f}" y2="{y_px(generated_mean):.2f}" stroke="#15803d" stroke-width="3" />'
        )
        labels.append(
            f'<text x="{center:.2f}" y="{pad_top + plot_height + 28}" text-anchor="middle" '
            f'font-family="sans-serif" font-size="13" fill="#0f172a">mb={cohort["minibatch_size"]}</text>'
        )
        labels.append(
            f'<text x="{native_center:.2f}" y="{pad_top + plot_height + 46}" text-anchor="middle" '
            f'font-family="sans-serif" font-size="12" fill="#2563eb">native {native_mean:.3g}</text>'
        )
        labels.append(
            f'<text x="{generated_center:.2f}" y="{pad_top + plot_height + 62}" text-anchor="middle" '
            f'font-family="sans-serif" font-size="12" fill="#15803d">gen {generated_mean:.3g}</text>'
        )

    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="{width}" height="{height}" fill="#f8fafc" />
<text x="{width/2:.0f}" y="28" text-anchor="middle" font-family="sans-serif" font-size="20" fill="#0f172a">{title}</text>
<text x="{width/2:.0f}" y="48" text-anchor="middle" font-family="sans-serif" font-size="13" fill="#475569">{subtitle}</text>
<text x="26" y="{height/2:.0f}" transform="rotate(-90 26 {height/2:.0f})" text-anchor="middle" font-family="sans-serif" font-size="14" fill="#334155">{y_label}</text>
<rect x="{pad_left}" y="{pad_top}" width="{plot_width}" height="{plot_height}" fill="#ffffff" stroke="#cbd5e1" stroke-width="1" />
{''.join(f'<line x1="{pad_left}" x2="{pad_left + plot_width}" y1="{y_px(tick):.2f}" y2="{y_px(tick):.2f}" stroke="#e2e8f0" stroke-width="1" />' for tick in y_ticks)}
{''.join(shapes)}
{''.join(f'<text x="{pad_left - 10}" y="{y_px(tick) + 4:.2f}" text-anchor="end" font-family="sans-serif" font-size="12" fill="#475569">{tick:.3g}</text>' for tick in y_ticks)}
{''.join(labels)}
<g transform="translate({pad_left + 20}, {pad_top + 18})">
  <circle cx="7" cy="7" r="6" fill="#2563eb" opacity="0.85" />
  <text x="20" y="11" font-family="sans-serif" font-size="12" fill="#0f172a">native</text>
  <circle cx="90" cy="7" r="6" fill="#16a34a" opacity="0.85" />
  <text x="103" y="11" font-family="sans-serif" font-size="12" fill="#0f172a">generated</text>
</g>
</svg>
"""


def _report_markdown(cohorts: list[dict[str, object]], output_dir: Path) -> str:
    lines = [
        "# Pong Heavy-Minibatch Cohort Report",
        "",
        "This bundle compares matched native-vs-generated replay cohorts for fixed heavy minibatch sizes after the loop-based Pong codegen change.",
        "",
    ]
    for cohort in cohorts:
        rows = cohort["rows"]
        sps_delta = [float(row["sps_delta"]) for row in rows]
        perf_delta = [float(row["perf_delta"]) for row in rows]
        return_delta = [float(row["episode_return_delta"]) for row in rows]
        lines.extend(
            [
                f"## Minibatch {cohort['minibatch_size']}",
                "",
                f"- pairs: `{len(rows)}`",
                f"- mean `SPS` delta: `{statistics.fmean(sps_delta):.3f}`",
                f"- median `SPS` delta: `{statistics.median(sps_delta):.3f}`",
                f"- mean `perf` delta: `{statistics.fmean(perf_delta):.6f}`",
                f"- median `perf` delta: `{statistics.median(perf_delta):.6f}`",
                f"- mean `episode_return` delta: `{statistics.fmean(return_delta):.6f}`",
                f"- median `episode_return` delta: `{statistics.median(return_delta):.6f}`",
                f"- cohort report: `{cohort['report_dir']}`",
                "",
            ]
        )
    lines.extend(
        [
            f"Artifacts live in `{output_dir}`.",
            "",
            "Use the cohort-specific `episode_return_ci.svg`, `perf_ci.svg`, and `sps_ci.svg` files for within-cohort training curves.",
        ]
    )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


SEASON_RE = re.compile(r"\b20\d{2}_\d{2}\b")

TRAIN_DIR = Path("ml/train")
FEATURES_DIR = Path("ml/features")
PREDICT_DIR = Path("ml/predict")

IMPORTANT_FILES = [
    "ml/features/export_features_v0.py",
    "ml/features/export_features_v2.py",
    "ml/features/export_features_v2_1.py",
    "ml/features/team_context.py",
    "ml/features/build_previous_season_player_priors.py",
    "ml/features/build_historical_staging_player_priors.py",
    "ml/train/train_ridge_rollform_v1.py",
    "ml/train/train_ridge_player_v2_1.py",
    "ml/train/train_gbr_player_v2_1.py",
    "ml/train/train_lgbm_player_v2_1.py",
    "ml/train/train_elasticnet_player_v2_1.py",
    "ml/train/eval_player_baselines_v2_1.py",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit current model training/feature code for multi-season readiness. "
            "This is static/read-only and does not modify the database."
        )
    )
    parser.add_argument(
        "--backend-root",
        default=".",
        help="Backend root containing ml/. Default: current directory.",
    )
    parser.add_argument(
        "--player-features-csv",
        default="",
        help="Optional player feature CSV exported by export_features_v2_1.",
    )
    parser.add_argument(
        "--historical-priors-csv",
        default="",
        help="Optional historical staging prior CSV from Day65.",
    )
    parser.add_argument("--out-json", required=True, help="Output JSON report path.")
    parser.add_argument("--out-md", required=True, help="Output Markdown report path.")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1")


def list_python_files(root: Path, subdir: Path) -> List[str]:
    directory = root / subdir
    if not directory.exists():
        return []
    return sorted(str(path.relative_to(root)) for path in directory.rglob("*.py"))


def line_findings(path: Path, patterns: Dict[str, str]) -> Dict[str, List[Dict[str, Any]]]:
    findings: Dict[str, List[Dict[str, Any]]] = {name: [] for name in patterns}
    if not path.exists():
        return findings

    lines = read_text(path).splitlines()
    for idx, line in enumerate(lines, start=1):
        for name, pattern in patterns.items():
            if re.search(pattern, line):
                findings[name].append({"line": idx, "text": line.strip()})
    return findings


def file_has(path: Path, text_fragment: str) -> bool:
    if not path.exists():
        return False
    return text_fragment in read_text(path)


def analyze_file(root: Path, relative_path: str) -> Dict[str, Any]:
    path = root / relative_path
    result: Dict[str, Any] = {
        "path": relative_path,
        "exists": path.exists(),
    }
    if not path.exists():
        return result

    text = read_text(path)
    seasons = sorted(set(SEASON_RE.findall(text)))

    patterns = {
        "hardcoded_season": r"\b20\d{2}_\d{2}\b",
        "groupby": r"\.groupby\(",
        "rolling": r"\.rolling\(",
        "shift": r"\.shift\(",
        "read_csv": r"pd\.read_csv|read_csv\(",
        "season_filter": r"season\s*=|WHERE .*season|s\.season|resolved_season|target_season|train_seasons",
        "player_id_reference": r"player_id",
        "raw_player_id_reference": r"raw_player_id",
        "train_seasons": r"train_seasons|--train-seasons",
        "start_end_gw_args": r"start_gw|--start-gw|--start_gw|end_gw|--end-gw|--end_gw",
    }

    findings = line_findings(path, patterns)

    result.update(
        {
            "line_count": len(text.splitlines()),
            "hardcoded_seasons": seasons,
            "has_groupby": bool(findings["groupby"]),
            "has_rolling": bool(findings["rolling"]),
            "has_shift": bool(findings["shift"]),
            "has_read_csv": bool(findings["read_csv"]),
            "has_season_filter_or_arg": bool(findings["season_filter"]),
            "has_train_seasons_arg": bool(findings["train_seasons"]),
            "has_start_end_gw_args": bool(findings["start_end_gw_args"]),
            "has_player_id_reference": bool(findings["player_id_reference"]),
            "has_raw_player_id_reference": bool(findings["raw_player_id_reference"]),
            "findings": findings,
        }
    )
    return result


def detect_groupby_scope_issues(file_report: Dict[str, Any]) -> List[Dict[str, Any]]:
    issues: List[Dict[str, Any]] = []
    findings = file_report.get("findings", {})
    groupby_lines = findings.get("groupby", [])

    for item in groupby_lines:
        text = item["text"]
        line = item["line"]

        if "player_id" in text and "season" not in text:
            issues.append(
                {
                    "severity": "warning",
                    "kind": "player_groupby_without_season_on_same_line",
                    "path": file_report["path"],
                    "line": line,
                    "text": text,
                    "message": (
                        "player_id groupby does not include season on the same line. "
                        "This may be safe only if the input dataframe is already filtered to one season."
                    ),
                }
            )

        if "team_id" in text and "season" not in text:
            issues.append(
                {
                    "severity": "warning",
                    "kind": "team_groupby_without_season_on_same_line",
                    "path": file_report["path"],
                    "line": line,
                    "text": text,
                    "message": (
                        "team_id groupby does not include season on the same line. "
                        "This may leak or mix form if a multi-season dataframe is passed."
                    ),
                }
            )

        if '"gw"' in text and "season" not in text and "[" not in text:
            issues.append(
                {
                    "severity": "info",
                    "kind": "gw_groupby_without_season_on_same_line",
                    "path": file_report["path"],
                    "line": line,
                    "text": text,
                    "message": (
                        "gw groupby does not include season on the same line. "
                        "Ranking or aggregation by gw should be season-scoped for multi-season data."
                    ),
                }
            )

    return issues


def audit_player_features_csv(path_str: str) -> Dict[str, Any]:
    if not path_str:
        return {"enabled": False}

    path = Path(path_str)
    result: Dict[str, Any] = {"enabled": True, "path": str(path), "exists": path.exists()}
    if not path.exists():
        result["errors"] = ["player features CSV does not exist."]
        return result

    df = pd.read_csv(path)
    columns = list(df.columns)
    result.update(
        {
            "shape": [int(df.shape[0]), int(df.shape[1])],
            "columns": columns,
            "has_season_column": "season" in columns,
            "has_player_id_column": "player_id" in columns,
            "has_raw_player_id_column": "raw_player_id" in columns,
            "has_prior_columns": any(col.startswith("prev_season_") for col in columns),
            "has_target_total_points": "total_points" in columns,
            "has_gw_column": "gw" in columns,
            "gw_min": int(df["gw"].min()) if "gw" in columns and len(df) else None,
            "gw_max": int(df["gw"].max()) if "gw" in columns and len(df) else None,
            "season_counts": df["season"].value_counts().to_dict() if "season" in columns else {},
            "duplicate_season_player_gw_rows": (
                int(df.duplicated(["season", "player_id", "gw"]).sum())
                if all(col in columns for col in ["season", "player_id", "gw"])
                else None
            ),
            "null_player_id_count": int(df["player_id"].isna().sum()) if "player_id" in columns else None,
            "null_season_count": int(df["season"].isna().sum()) if "season" in columns else None,
        }
    )
    return result


def audit_historical_priors_csv(path_str: str) -> Dict[str, Any]:
    if not path_str:
        return {"enabled": False}

    path = Path(path_str)
    result: Dict[str, Any] = {"enabled": True, "path": str(path), "exists": path.exists()}
    if not path.exists():
        result["errors"] = ["historical priors CSV does not exist."]
        return result

    df = pd.read_csv(path)
    columns = list(df.columns)
    result.update(
        {
            "shape": [int(df.shape[0]), int(df.shape[1])],
            "columns": columns,
            "has_source_season": "source_season" in columns,
            "has_target_season": "target_season" in columns,
            "has_raw_player_id": "raw_player_id" in columns,
            "has_canonical_player_id": "canonical_player_id" in columns,
            "all_canonical_player_id_null": (
                bool(df["canonical_player_id"].isna().all()) if "canonical_player_id" in columns else None
            ),
            "source_season_counts": df["source_season"].value_counts().to_dict() if "source_season" in columns else {},
            "target_season_counts": df["target_season"].value_counts().to_dict() if "target_season" in columns else {},
            "prior_identity_scope_counts": (
                df["prior_identity_scope"].value_counts().to_dict()
                if "prior_identity_scope" in columns
                else {}
            ),
            "prior_source_counts": df["prior_source"].value_counts().to_dict() if "prior_source" in columns else {},
            "row_count": int(len(df)),
            "active_player_count": int((df["prev_season_minutes"] > 0).sum())
            if "prev_season_minutes" in columns
            else None,
            "total_prev_season_minutes": int(df["prev_season_minutes"].sum())
            if "prev_season_minutes" in columns
            else None,
            "total_prev_season_points": int(df["prev_season_total_points"].sum())
            if "prev_season_total_points" in columns
            else None,
        }
    )
    return result


def build_findings(root: Path, file_reports: List[Dict[str, Any]], player_csv: Dict[str, Any], priors_csv: Dict[str, Any]) -> List[Dict[str, Any]]:
    findings: List[Dict[str, Any]] = []

    report_by_path = {report["path"]: report for report in file_reports}

    # Day65 prior status.
    if priors_csv.get("enabled") and priors_csv.get("exists"):
        if priors_csv.get("all_canonical_player_id_null") is True:
            findings.append(
                {
                    "severity": "blocker",
                    "area": "identity_mapping",
                    "title": "Historical priors cannot be joined to canonical players yet.",
                    "details": (
                        "Day65 priors use raw_player_id and all canonical_player_id values are null. "
                        "This is safe for staging but blocks direct model-feature integration."
                    ),
                }
            )

    # Current feature CSV status.
    if player_csv.get("enabled") and player_csv.get("exists"):
        if player_csv.get("has_prior_columns") is False:
            findings.append(
                {
                    "severity": "info",
                    "area": "player_features",
                    "title": "Current player feature CSV has no previous-season prior columns.",
                    "details": (
                        "The current export_features_v2_1 output is current-season rolling/context only. "
                        "Previous-season priors are not integrated yet."
                    ),
                }
            )
        if player_csv.get("has_season_column") is True and player_csv.get("has_player_id_column") is True:
            findings.append(
                {
                    "severity": "pass",
                    "area": "player_features",
                    "title": "Player feature artifact includes season and player_id.",
                    "details": (
                        "The exported CSV can be audited at season/player/GW grain. "
                        "Rows should still be generated per season until feature exporters are made multi-season-safe."
                    ),
                }
            )

    # Static groupby issues.
    for report in file_reports:
        findings.extend(detect_groupby_scope_issues(report))

    # Key file-specific conclusions.
    ridge_rollform = report_by_path.get("ml/train/train_ridge_rollform_v1.py", {})
    if ridge_rollform.get("exists"):
        if ridge_rollform.get("has_train_seasons_arg"):
            findings.append(
                {
                    "severity": "partial_pass",
                    "area": "training",
                    "title": "train_ridge_rollform_v1 has explicit train_seasons support.",
                    "details": (
                        "This is the most multi-season-aware player training path, but it reads from canonical "
                        "player_gw_stats, so it cannot consume Day64 staging data until mapping/import is resolved."
                    ),
                }
            )

    export_v2_1 = report_by_path.get("ml/features/export_features_v2_1.py", {})
    if export_v2_1.get("exists"):
        if export_v2_1.get("has_start_end_gw_args"):
            findings.append(
                {
                    "severity": "pass",
                    "area": "feature_export",
                    "title": "export_features_v2_1 requires explicit GW range.",
                    "details": (
                        "Day66 confirmed it must be run with --start-gw/--end-gw or --start_gw/--end_gw."
                    ),
                }
            )
        if export_v2_1.get("has_season_filter_or_arg"):
            findings.append(
                {
                    "severity": "partial_pass",
                    "area": "feature_export",
                    "title": "export_features_v2_1 is season-filtered.",
                    "details": (
                        "It exports one season at a time and appends a season column. "
                        "Its rolling groupby is by player_id because the input is filtered to a single season."
                    ),
                }
            )

    team_context = report_by_path.get("ml/features/team_context.py", {})
    if team_context.get("exists"):
        if team_context.get("has_groupby"):
            findings.append(
                {
                    "severity": "warning",
                    "area": "team_context",
                    "title": "team_context rolling logic appears team_id-scoped, not season-scoped.",
                    "details": (
                        "This is safe only when called with one season at a time. "
                        "For true multi-season feature generation, add season-aware grouping."
                    ),
                }
            )

    csv_trainers = [
        report
        for report in file_reports
        if report["path"].startswith("ml/train/")
        and report.get("has_read_csv")
        and not report.get("has_train_seasons_arg")
    ]
    if csv_trainers:
        findings.append(
            {
                "severity": "warning",
                "area": "training",
                "title": "Most v2_1 model trainers consume pre-exported CSVs and do not own season logic.",
                "details": (
                    "These trainers can train on multi-season data only if the CSV artifact is already safely "
                    "constructed. They should not be considered multi-season-safe by themselves."
                ),
                "files": [report["path"] for report in csv_trainers],
            }
        )

    return findings


def severity_counts(findings: List[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for finding in findings:
        severity = finding.get("severity", "unknown")
        counts[severity] = counts.get(severity, 0) + 1
    return counts


def build_report(args: argparse.Namespace) -> Dict[str, Any]:
    root = Path(args.backend_root).resolve()

    train_files = list_python_files(root, TRAIN_DIR)
    feature_files = list_python_files(root, FEATURES_DIR)
    predict_files = list_python_files(root, PREDICT_DIR)

    files_to_analyze = sorted(set(IMPORTANT_FILES + train_files + feature_files + predict_files))
    file_reports = [analyze_file(root, rel_path) for rel_path in files_to_analyze]

    player_csv = audit_player_features_csv(args.player_features_csv)
    priors_csv = audit_historical_priors_csv(args.historical_priors_csv)

    findings = build_findings(root, file_reports, player_csv, priors_csv)

    blockers = [finding for finding in findings if finding.get("severity") == "blocker"]

    report = {
        "created_at": utc_now(),
        "backend_root": str(root),
        "passed": len(blockers) == 0,
        "multi_season_model_training_ready": False,
        "day66_scope": "audit_only",
        "summary": {
            "train_file_count": len(train_files),
            "feature_file_count": len(feature_files),
            "predict_file_count": len(predict_files),
            "analyzed_file_count": len(file_reports),
            "severity_counts": severity_counts(findings),
        },
        "artifacts": {
            "player_features_csv": player_csv,
            "historical_priors_csv": priors_csv,
        },
        "files": file_reports,
        "findings": findings,
        "recommended_next_steps": [
            "Keep Day65 priors as staging-only until historical player identity mapping is resolved.",
            "Do not directly merge Day65 raw_player_id priors into canonical player_id feature exports.",
            "Create a player identity mapping audit/template before model integration.",
            "If generating multi-season CSVs, update player rolling features to group by season + player_id.",
            "If generating multi-season team features, update team_context rolling features to group by season + team_id.",
            "For current-season-only exports, continue running export_features_v2_1 one season at a time with explicit GW bounds.",
        ],
    }

    return report


def write_json(report: Dict[str, Any], path: str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True, default=str), encoding="utf-8")


def format_finding(finding: Dict[str, Any]) -> str:
    severity = finding.get("severity", "unknown")
    area = finding.get("area") or finding.get("kind") or "general"
    title = finding.get("title") or finding.get("message") or "Finding"
    details = finding.get("details") or finding.get("text") or ""

    parts = ["- **[%s] %s** — %s" % (severity.upper(), area, title)]
    if details:
        parts.append("  - %s" % details)
    if finding.get("path"):
        location = finding["path"]
        if finding.get("line"):
            location += ":%s" % finding["line"]
        parts.append("  - Location: `%s`" % location)
    if finding.get("files"):
        parts.append("  - Files: `%s`" % "`, `".join(finding["files"]))
    return "\n".join(parts)


def write_markdown(report: Dict[str, Any], path: str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    player_csv = report["artifacts"]["player_features_csv"]
    priors_csv = report["artifacts"]["historical_priors_csv"]

    lines: List[str] = []
    lines.append("# Day66 Multi-Season Model Training Audit")
    lines.append("")
    lines.append("Created at: `%s`" % report["created_at"])
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("- Backend root: `%s`" % report["backend_root"])
    lines.append("- Scope: `%s`" % report["day66_scope"])
    lines.append("- Passed: `%s`" % report["passed"])
    lines.append("- Multi-season model training ready: `%s`" % report["multi_season_model_training_ready"])
    lines.append("- Severity counts: `%s`" % report["summary"]["severity_counts"])
    lines.append("")

    lines.append("## Artifact Audit")
    lines.append("")
    lines.append("### Player Features CSV")
    lines.append("")
    lines.append("- Enabled: `%s`" % player_csv.get("enabled"))
    lines.append("- Exists: `%s`" % player_csv.get("exists"))
    if player_csv.get("exists"):
        lines.append("- Shape: `%s`" % player_csv.get("shape"))
        lines.append("- Has season column: `%s`" % player_csv.get("has_season_column"))
        lines.append("- Has player_id column: `%s`" % player_csv.get("has_player_id_column"))
        lines.append("- Has prior columns: `%s`" % player_csv.get("has_prior_columns"))
        lines.append("- GW range: `%s` to `%s`" % (player_csv.get("gw_min"), player_csv.get("gw_max")))
        lines.append("- Season counts: `%s`" % player_csv.get("season_counts"))
    lines.append("")

    lines.append("### Historical Priors CSV")
    lines.append("")
    lines.append("- Enabled: `%s`" % priors_csv.get("enabled"))
    lines.append("- Exists: `%s`" % priors_csv.get("exists"))
    if priors_csv.get("exists"):
        lines.append("- Shape: `%s`" % priors_csv.get("shape"))
        lines.append("- Has raw_player_id: `%s`" % priors_csv.get("has_raw_player_id"))
        lines.append("- Has canonical_player_id: `%s`" % priors_csv.get("has_canonical_player_id"))
        lines.append("- All canonical_player_id null: `%s`" % priors_csv.get("all_canonical_player_id_null"))
        lines.append("- Row count: `%s`" % priors_csv.get("row_count"))
        lines.append("- Active player count: `%s`" % priors_csv.get("active_player_count"))
        lines.append("- Total prev-season minutes: `%s`" % priors_csv.get("total_prev_season_minutes"))
        lines.append("- Total prev-season points: `%s`" % priors_csv.get("total_prev_season_points"))
        lines.append("- Prior identity scopes: `%s`" % priors_csv.get("prior_identity_scope_counts"))
    lines.append("")

    lines.append("## Findings")
    lines.append("")
    for finding in report["findings"]:
        lines.append(format_finding(finding))
        lines.append("")

    lines.append("## Recommended Next Steps")
    lines.append("")
    for step in report["recommended_next_steps"]:
        lines.append("- %s" % step)
    lines.append("")

    out.write_text("\n".join(lines), encoding="utf-8")


def print_summary(report: Dict[str, Any], out_json: str, out_md: str) -> None:
    print("=== Day66 Multi-Season Model Training Audit ===")
    print("passed:", report["passed"])
    print("multi_season_model_training_ready:", report["multi_season_model_training_ready"])
    print("severity_counts:", report["summary"]["severity_counts"])
    print("train_file_count:", report["summary"]["train_file_count"])
    print("feature_file_count:", report["summary"]["feature_file_count"])
    print("predict_file_count:", report["summary"]["predict_file_count"])
    print("saved_json:", out_json)
    print("saved_md:", out_md)
    print()
    print("Top findings:")
    for finding in report["findings"][:12]:
        severity = finding.get("severity", "unknown").upper()
        area = finding.get("area") or finding.get("kind") or "general"
        title = finding.get("title") or finding.get("message") or "Finding"
        print("- [%s] %s: %s" % (severity, area, title))


def main() -> None:
    args = parse_args()
    report = build_report(args)
    write_json(report, args.out_json)
    write_markdown(report, args.out_md)
    print_summary(report, args.out_json, args.out_md)


if __name__ == "__main__":
    main()

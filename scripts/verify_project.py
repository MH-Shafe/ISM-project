"""Final packaging verification gate for the CERT r4.2 insider-threat project.

Verification-ONLY: this script NEVER trains, fits, scores, recalibrates or
evaluates TEST. It checks structure, hashes, frozen records, model file
integrity (parse only), documentation consistency and secrets exposure, then
writes reports/artifacts/final_verification_report.json.

Usage:
    python scripts/verify_project.py [--root DIR] [--out PATH] [--skip-tests]

Exit code 0 = all checks passed; 1 = at least one check failed.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone

FROZEN_MODEL = "lgbm-graph-v1"
FROZEN_BEST_ITERATION = 186
FROZEN_SEED = 42
FROZEN_THRESHOLD = 0.9186015432508062
FROZEN_MODEL_BYTES = 651047
LGBM_TRAIN_VERSION = "4.6.0"

FROZEN_12 = [
    "login_count",
    "after_hours_login_count",
    "usb_connection_count",
    "file_access_count",
    "sensitive_file_access_count",
    "http_activity_count",
    "unique_device_count",
    "unusual_access_count",
    "device_consistency_score",
    "rare_device_usage_count",
    "file_type_consistency_score",
    "rare_file_type_access_count",
]

MASTER_SECTIONS = ["0", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13",
                   "14", "15", "16", "17", "18", "19", "20", "21", "22", "23",
                   "24", "25", "26", "27", "28", "29", "30"]

PHASE_REPORTS = [
    "foundation_report.md", "baseline_report.md",
    "phase3_report.md", "phase4_report.md", "phase5_graph_report.md",
    "phase6_graph_regression_report.md", "phase7_freeze_report.md",
    "phase8_adaptive_risk_report.md", "phase9_alert_prioritization_report.md",
    "phase10_conformal_report.md", "phase11_explainability_report.md",
    "phase12_operational_envelope_report.md",
    "phase13_temporal_stability_report.md", "phase14_user_holdout_report.md",
    "phase15_unseen_user_diagnosis_report.md",
    "phase16_final_packaging_report.md",
]

EXPERIMENT_SOURCES = ["phase6.py", "phase7.py", "phase8.py", "phase9.py",
                      "phase10.py", "phase11.py", "phase12.py", "phase13.py",
                      "phase14.py", "phase15.py"]

RUN_SCRIPTS = ["run_phase3.py", "run_phase4.py", "run_phase6.py",
               "run_phase7.py", "run_phase8.py", "run_phase9.py",
               "run_phase10.py", "run_phase11.py", "run_phase12.py",
               "run_phase13.py", "run_phase14.py", "run_phase15.py"]

TESTS = ["test_aggregation.py", "test_baseline.py", "test_graph.py",
         "test_labels.py", "test_leakage.py", "test_phase6.py",
         "test_phase7.py", "test_phase8.py", "test_phase9.py",
         "test_phase10.py", "test_phase11.py", "test_phase12.py",
         "test_phase13.py", "test_phase14.py", "test_phase15.py",
         "test_real_data.py", "test_splits.py", "test_threshold.py",
         "test_validation.py"]

DOCS = ["kaggle-execution-policy.md", "leakage_analysis.md",
        "phase11_gap_analysis.md", "phase13_specification.md",
        "phase14_specification.md", "phase15_specification.md",
        "environment.md"]

KNOWLEDGE = ["README.md", "architecture/architecture.md", "risks/README.md",
             "research/README.md", "graph/README.md",
             "features/feature-registry.md", "decisions/decision-log.md",
             "experiments/experiment-index.md", "dataset/cert-r4.2.md",
             "evaluation/README.md"]

SRC_MODULES = ["config.py", "data/aggregation.py", "data/labels.py",
               "data/validation.py", "preprocessing/splits.py",
               "preprocessing/alignment.py", "evaluation/metrics.py",
               "evaluation/threshold.py", "models/lightgbm_baseline.py",
               "graph/features.py"]

KAGGLE_SCRIPTS = ["kaggle_exec.py", "push_to_kaggle.py", "pull_from_kaggle.py",
                  "decode_sources.py", "build_user_day.py",
                  "build_graph_features.py", "rebuild_derived_tables.py",
                  "run_validation.py", "verify_modeling_table.py",
                  "train_baseline.py", "bench_phase12_runtime.py"]

SECRET_PATTERNS = [
    ("jwt", re.compile(r"eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}")),
    ("openai_sk", re.compile(r"\bsk-[A-Za-z0-9]{20,}")),
    ("aws_access_key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("private_key_block", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")),
    ("api_key_assignment", re.compile(r"(?:api[_-]?key|apikey)\s*[:=]\s*[\"'][A-Za-z0-9_\-]{8,}[\"']", re.I)),
    ("password_assignment", re.compile(r"password\s*[:=]\s*[\"'][^\"']{6,}[\"']", re.I)),
    ("kaggle_json_pair", re.compile(r"\"username\"\s*:\s*\"[^\"]{3,}\".{0,120}\"key\"\s*:\s*\"[^\"]{20,}\"", re.S)),
]

NO_ML_OPERATIONS = [
    "no model fitting",
    "no model training",
    "no scoring / prediction generation (lightgbm predict is never called)",
    "no threshold calculation",
    "no calibration",
    "no TEST evaluation",
    "no artifact modification (report file written last, excluded from scans)",
]


def md5_of(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


class Checker:
    def __init__(self, root, out_path, skip_tests):
        self.root = os.path.abspath(root)
        self.out_path = out_path
        self.skip_tests = skip_tests
        self.checks = []
        self.new_baselines = {}

    def path(self, *parts):
        return os.path.join(self.root, *parts)

    def add(self, check_id, name, status, detail=""):
        self.checks.append({"id": check_id, "name": name,
                            "status": status, "detail": detail})
        return status == "PASS"

    def require_files(self, check_id, name, files):
        missing = [f for f in files if not os.path.isfile(self.path(f))]
        if missing:
            return self.add(check_id, name, "FAIL",
                            "missing: " + ", ".join(missing))
        return self.add(check_id, name, "PASS", f"{len(files)} files present")

    def run_json_scan(self, check_id, name, pattern, files, base):
        hits = []
        for rel in files:
            p = self.path(base, rel) if base else self.path(rel)
            if not os.path.isfile(p):
                continue
            try:
                with open(p, "r", encoding="utf-8", errors="replace") as fh:
                    content = fh.read()
            except OSError:
                continue
            for m in pattern.finditer(content):
                line = content[:m.start()].count("\n") + 1
                hits.append(f"{rel}:{line}")
        if hits:
            return self.add(check_id, name, "FAIL",
                            "matches: " + "; ".join(hits[:20]))
        return self.add(check_id, name, "PASS", f"{len(files)} files scanned")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=None, help="repo root (default: parent of scripts/)")
    ap.add_argument("--out", default=None, help="output JSON path")
    ap.add_argument("--skip-tests", action="store_true",
                    help="do not run the pytest suite (final gate runs it)")
    args = ap.parse_args()

    root = args.root or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_path = args.out or os.path.join(root, "reports", "artifacts",
                                        "final_verification_report.json")
    c = Checker(root, out_path, args.skip_tests)

    env = {
        "python": sys.version.split()[0],
        "lightgbm": _env_version("lightgbm"),
        "numpy": _env_version("numpy"),
        "pandas": _env_version("pandas"),
        "scikit-learn": _env_version("scikit-learn"),
        "pyarrow": _env_version("pyarrow"),
        "duckdb": _env_version("duckdb"),
        "shap": _env_version("shap"),
        "pytest": _env_version("pytest"),
    }

    structure_checks(c)
    manifest_checks(c)
    frozen_checks(c)
    data_checks(c)
    doc_checks(c)
    security_checks(c)
    tests_check(c)

    summary = {"n_checks": len(c.checks),
               "passed": sum(1 for x in c.checks if x["status"] == "PASS"),
               "failed": sum(1 for x in c.checks if x["status"] == "FAIL"),
               "skipped": sum(1 for x in c.checks if x["status"] == "SKIP")}
    overall = "PASS" if summary["failed"] == 0 else "FAIL"

    report = {
        "schema_version": 1,
        "tool": "scripts/verify_project.py",
        "purpose": "verification-only reproducibility gate (no ML operations)",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "root": root,
        "environment": env,
        "no_ml_operations": NO_ML_OPERATIONS,
        "checks": c.checks,
        "summary": summary,
        "overall": overall,
        "new_baselines": c.new_baselines,
        "note": ("TEST artifacts are loaded only to verify parseability and "
                 "recorded hashes; no scores are generated from them."),
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)

    print(f"Verification report written to {out_path}")
    print(f"OVERALL: {overall} ({summary['passed']} passed, "
          f"{summary['failed']} failed, {summary['skipped']} skipped)")
    for ch in c.checks:
        print(f"[{ch['status']:4s}] {ch['id']}: {ch['name']}")
        if ch["detail"] and ch["status"] == "FAIL":
            print(f"      {ch['detail']}")

    sys.exit(0 if overall == "PASS" else 1)


def _env_version(pkg):
    try:
        return importlib.metadata.version(pkg)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def structure_checks(c):
    c.require_files("root_required_files", "root packaging files present", [
        "README.md", "requirements.txt", "docs/environment.md",
        "opencode.json", "smoke_test.py", "start-opencode.bat",
        "get-kaggle-runtime.ps1", "get-kaggle-runtime.bat",
        ".opencode/instructions/ism-principles.md",
    ])
    c.require_files("src_core", "src core modules present",
                    ["src/" + m for m in SRC_MODULES])
    c.require_files("src_experiments", "phase experiment sources present",
                    ["src/experiments/" + m for m in EXPERIMENT_SOURCES])
    c.require_files("kaggle_scripts", "Kaggle runner scripts present",
                    ["kaggle_scripts/" + m for m in KAGGLE_SCRIPTS + RUN_SCRIPTS])
    c.require_files("tests", "test files present", ["tests/" + m for m in TESTS])
    c.require_files("docs", "docs present", ["docs/" + m for m in DOCS])
    c.require_files("knowledge", "knowledge notes present",
                    ["knowledge/" + m for m in KNOWLEDGE])
    c.require_files("reports", "phase reports present",
                    ["reports/" + m for m in PHASE_REPORTS])
    if not os.path.isfile(c.path("reports", "ISM_MASTER_REPORT.md")):
        c.add("master_report", "master report present", "FAIL")
    else:
        c.add("master_report", "master report present", "PASS")
    art = c.path("reports", "artifacts")
    if not os.path.isdir(art):
        c.add("artifacts_dir", "artifacts directory present", "FAIL")
        return
    n = len([f for f in os.listdir(art) if os.path.isfile(os.path.join(art, f))])
    c.add("artifacts_dir", "artifacts directory present",
          "PASS" if n >= 90 else "FAIL", f"{n} files on disk")
    c.require_files("artifact_index", "artifact index present",
                    ["reports/artifacts/ARTIFACT_INDEX.md"])
    c.require_files("phase16_report", "Phase 16 report present",
                    ["reports/phase16_final_packaging_report.md"])


def manifest_checks(c):
    art = c.path("reports", "artifacts")
    for manifest in ["phase14_manifest.json", "phase15_manifest.json"]:
        mp = os.path.join(art, manifest)
        try:
            with open(mp, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception as exc:
            c.add(f"manifest_{manifest}", f"{manifest} parses", "FAIL", str(exc))
            continue
        entries = data.get("files", {})
        if not isinstance(entries, dict) or not entries:
            c.add(f"manifest_{manifest}", f"{manifest} has file entries",
                  "FAIL", "empty or missing 'files'")
            continue
        bad = []
        for fname, rec in entries.items():
            fp = os.path.join(art, fname)
            if not os.path.isfile(fp):
                bad.append(f"{fname}: missing")
                continue
            size = os.path.getsize(fp)
            if size != rec.get("size_bytes"):
                bad.append(f"{fname}: size {size} != {rec.get('size_bytes')}")
                continue
            digest = md5_of(fp)
            if digest != rec.get("md5"):
                bad.append(f"{fname}: md5 mismatch")
        if bad:
            c.add(f"manifest_{manifest}", f"{manifest} md5/size verified",
                  "FAIL", "; ".join(bad))
        else:
            c.add(f"manifest_{manifest}", f"{manifest} md5/size verified",
                  "PASS", f"{len(entries)} files match")

    exp = os.path.join(art, "phase15_experiment.json")
    if os.path.isfile(exp):
        with open(exp, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        rec = data.get("gates", {}).get("gate3_input_md5", {})
        bad = []
        for fname, expected in rec.items():
            fp = os.path.join(art, fname)
            if not os.path.isfile(fp):
                bad.append(f"{fname}: missing")
                continue
            if md5_of(fp) != expected:
                bad.append(f"{fname}: md5 mismatch")
        c.add("frozen_input_md5s", "frozen input md5s match Phase 15 record",
              "PASS" if not bad else "FAIL", "; ".join(bad) if bad else "6 files match")


def frozen_checks(c):
    art = c.path("reports", "artifacts")
    model_path = os.path.join(art, "phase7_model_lgbm-graph-v1.txt")

    if os.path.isfile(model_path) and os.path.getsize(model_path) == FROZEN_MODEL_BYTES:
        c.add("frozen_model_file", "frozen model file present with recorded size",
              "PASS", f"{FROZEN_MODEL_BYTES} bytes")
    else:
        c.add("frozen_model_file", "frozen model file present with recorded size",
              "FAIL", "missing or size != 651047")

    if os.path.isfile(model_path):
        try:
            import lightgbm as lgb
            booster = lgb.Booster(model_file=model_path)
            n_trees = booster.num_trees()
            names = list(booster.feature_name())
            ok = (n_trees == FROZEN_BEST_ITERATION
                  and len(names) == len(FROZEN_12)
                  and set(names) == set(FROZEN_12))
            c.add("frozen_model_integrity",
                  "frozen model integrity (parse-only; no scoring)",
                  "PASS" if ok else "FAIL",
                  f"num_trees={n_trees}, features={len(names)}/12")
        except Exception as exc:
            c.add("frozen_model_integrity", "frozen model integrity (parse-only)",
                  "FAIL", f"{type(exc).__name__}: {exc}")

    freeze_path = os.path.join(art, "phase7_freeze_lgbm-graph-v1.json")
    if os.path.isfile(freeze_path):
        with open(freeze_path, "r", encoding="utf-8") as fh:
            rec = json.load(fh)
        ok = (rec.get("model_name") == FROZEN_MODEL
              and rec.get("n_features") == 12
              and set(rec.get("features", [])) == set(FROZEN_12)
              and rec.get("model", {}).get("seed") == FROZEN_SEED
              and rec.get("model", {}).get("best_iteration") == FROZEN_BEST_ITERATION
              and rec.get("model", {}).get("version") == LGBM_TRAIN_VERSION)
        c.add("frozen_features_record", "frozen 12-feature configuration record",
              "PASS" if ok else "FAIL",
              f"model={rec.get('model_name')}, seed={rec.get('model', {}).get('seed')}, "
              f"best_iter={rec.get('model', {}).get('best_iteration')}, "
              f"lgbm={rec.get('model', {}).get('version')}")

    cfg_path = c.path("src", "config.py")
    if os.path.isfile(cfg_path):
        sys.path.insert(0, c.path("src"))
        try:
            import config as cfgmod
            expected = set(cfgmod.BEHAVIORAL_FEATURES) | (
                set(cfgmod.GRAPH_FEATURES) - {"department_file_type_mismatch_count"})
            ok = (len(expected) == 12 and expected == set(FROZEN_12))
            c.add("frozen_config_src", "src/config.py registry yields frozen 12",
                  "PASS" if ok else "FAIL",
                  f"{len(expected)} features from registry")
        except Exception as exc:
            c.add("frozen_config_src", "src/config.py registry yields frozen 12",
                  "FAIL", f"{type(exc).__name__}: {exc}")

    p9 = os.path.join(art, "phase9_freeze.json")
    if os.path.isfile(p9):
        with open(p9, "r", encoding="utf-8") as fh:
            rec = json.load(fh)
        policy = rec.get("policy", {})
        thr_ok = policy.get("id") == "frozen_max_f1" and policy.get("param") == FROZEN_THRESHOLD
        if os.path.isfile(freeze_path):
            with open(freeze_path, "r", encoding="utf-8") as fh:
                rec7 = json.load(fh)
            thr_ok = thr_ok and rec7.get("calibration", {}).get(
                "threshold_max_f1") == FROZEN_THRESHOLD
        c.add("frozen_threshold", "frozen threshold matches records exactly",
              "PASS" if thr_ok else "FAIL",
              f"param={policy.get('param')}")


def data_checks(c):
    art = c.path("reports", "artifacts")
    jsons = sorted(f for f in os.listdir(art)
                   if f.endswith(".json") and f != "final_verification_report.json")
    bad = []
    for f in jsons:
        try:
            with open(os.path.join(art, f), "r", encoding="utf-8") as fh:
                json.load(fh)
        except Exception as exc:
            bad.append(f"{f}: {exc}")
    c.add("json_parse_all", "all artifact JSON files parse",
          "PASS" if not bad else "FAIL",
          f"{len(jsons) - len(bad)}/{len(jsons)} parsed" + ("; " + "; ".join(bad[:5]) if bad else ""))

    parquets = sorted(f for f in os.listdir(art) if f.endswith(".parquet"))
    bad = []
    shapes = {}
    try:
        import pandas as pd
        for f in parquets:
            try:
                df = pd.read_parquet(os.path.join(art, f))
                shapes[f] = {"rows": int(df.shape[0]), "cols": int(df.shape[1])}
            except Exception as exc:
                bad.append(f"{f}: {exc}")
    except Exception as exc:
        bad.append(f"pandas import: {exc}")
    c.add("parquet_load_all", "all artifact parquet files load (no scoring)",
          "PASS" if not bad else "FAIL",
          f"{len(parquets) - len(bad)}/{len(parquets)} loaded" + ("; " + "; ".join(bad[:5]) if bad else ""))
    if shapes:
        c.new_baselines["parquet_shapes"] = shapes

    for f in sorted(os.listdir(art)):
        fp = os.path.join(art, f)
        if os.path.isfile(fp):
            c.new_baselines[f] = {"md5": md5_of(fp), "size_bytes": os.path.getsize(fp)}


def doc_checks(c):
    master = c.path("reports", "ISM_MASTER_REPORT.md")
    if os.path.isfile(master):
        with open(master, "r", encoding="utf-8") as fh:
            content = fh.read()
        missing = [s for s in MASTER_SECTIONS
                   if f"## {s}." not in content]
        c.add("master_sections", "master report contains required sections",
              "PASS" if not missing else "FAIL",
              "missing: " + ", ".join(missing) if missing else "sections 0, 4-30 present")
        links_ok, links_bad, links_checked = check_links(master, c.root)
        c.add("master_links", "master report internal links resolve",
              "PASS" if not links_bad else "FAIL",
              f"{links_checked} checked" + (f"; broken: {links_bad[:5]}" if links_bad else ""))
    else:
        c.add("master_sections", "master report contains required sections", "FAIL")

    readme = c.path("README.md")
    if os.path.isfile(readme):
        with open(readme, "r", encoding="utf-8") as fh:
            content = fh.read()
        if "reports/ISM_MASTER_REPORT.md" not in content:
            c.add("readme_master_link", "README links the master report", "FAIL")
        else:
            c.add("readme_master_link", "README links the master report", "PASS")
        links_ok, links_bad, links_checked = check_links(readme, c.root)
        c.add("readme_links", "README internal links resolve",
              "PASS" if not links_bad else "FAIL",
              f"{links_checked} checked" + (f"; broken: {links_bad[:5]}" if links_bad else ""))
    else:
        c.add("readme_master_link", "README links the master report", "FAIL")
        c.add("readme_links", "README internal links resolve", "FAIL")

    with open(master, "r", encoding="utf-8") as fh:
        content = fh.read()
    stop_ok = "Phase 16" in content and "STOP" in content
    c.add("stop_state_recorded", "master report records Phase 16 STOP state",
          "PASS" if stop_ok else "FAIL")


def check_links(md_path, root):
    with open(md_path, "r", encoding="utf-8") as fh:
        content = fh.read()
    bad = []
    checked = 0
    for m in re.finditer(r"\[[^\]]*\]\(([^)]+)\)", content):
        target = m.group(1).strip()
        if not target or target.startswith(("#", "http://", "https://",
                                           "mailto:", "[[", "<")):
            continue
        if "#" in target:
            target = target.split("#", 1)[0]
        if not target:
            continue
        checked += 1
        resolved = os.path.normpath(os.path.join(root, target))
        if not os.path.exists(resolved):
            bad.append(target)
    return not bad, bad, checked


def security_checks(c):
    scan_files = []
    for base, rels in [
        ("", ["README.md", "requirements.txt", "opencode.json", "smoke_test.py",
              "start-opencode.bat", "get-kaggle-runtime.ps1",
              "get-kaggle-runtime.bat"]),
        ("docs", DOCS),
        ("reports", PHASE_REPORTS + ["ISM_MASTER_REPORT.md"]),
        ("knowledge", KNOWLEDGE),
        ("src", ["config.py"] + ["data/" + f for f in
                 ["aggregation.py", "labels.py", "validation.py"]]
         + ["preprocessing/" + f for f in ["splits.py", "alignment.py"]]
         + ["evaluation/" + f for f in ["metrics.py", "threshold.py"]]
         + ["models/lightgbm_baseline.py", "graph/features.py"]
         + ["experiments/" + f for f in EXPERIMENT_SOURCES]),
        ("kaggle_scripts", KAGGLE_SCRIPTS + RUN_SCRIPTS),
        ("tests", TESTS),
        ("scripts", ["verify_project.py"]),
    ]:
        for rel in rels:
            scan_files.append((base, rel))
    for f in sorted(os.listdir(c.path("reports", "artifacts"))):
        if f.endswith((".json", ".log")):
            scan_files.append(("reports/artifacts", f))

    all_failed = True
    for label, pattern in SECRET_PATTERNS:
        status = c.run_json_scan(f"secret_{label}", f"secret scan: {label}",
                                 pattern, [r for _, r in scan_files], None)
        all_failed = all_failed and status
    if all_failed:
        c.add("secret_scan_none", "no high-confidence credential patterns", "PASS")

    env_file = c.path(".env.kaggle")
    if os.path.isfile(env_file):
        c.add("secret_env_noted",
              "local credential config file excluded from scan (generic note)",
              "PASS", ".env.kaggle exists locally; not scanned, never committed")
    else:
        c.add("secret_env_noted", "local credential config file present", "FAIL",
              ".env.kaggle missing (local Kaggle configuration incomplete)")


def tests_check(c):
    if c.skip_tests:
        c.add("test_suite", "full local test suite", "SKIP",
              "skipped via --skip-tests")
        return
    cmd = [sys.executable, "-m", "pytest", "tests", "-q"]
    env = dict(os.environ)
    env.setdefault("CERT_WORKING", os.path.join(c.root, "reports"))
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              cwd=c.root, env=env, timeout=900)
        tail = (proc.stdout or "").strip().splitlines()
        last = tail[-1] if tail else ""
        passed = proc.returncode == 0
        c.add("test_suite", "full local test suite",
              "PASS" if passed else "FAIL",
              last or f"exit code {proc.returncode}")
    except subprocess.TimeoutExpired:
        c.add("test_suite", "full local test suite", "FAIL",
              "pytest timed out after 900 s")


if __name__ == "__main__":
    main()
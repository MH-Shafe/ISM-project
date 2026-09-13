"""Central configuration for the CERT insider-threat project.

All dataset paths, schemas, formats and defaults live here so that
validation, aggregation and evaluation code stays data-driven.
"""
import os
import re

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# Authoritative dataset location (Kaggle mount). Overridable via env var so
# tests can run against synthetic data trees.
CERT_DATA_ROOT = os.environ.get(
    "CERT_DATA_ROOT",
    "/kaggle/input/datasets/andrihjonior/cert-insider-threat-dataset-r4-2",
)
R4_2_DIR = os.path.join(CERT_DATA_ROOT, "r4.2")
ANSWERS_DIR = os.path.join(CERT_DATA_ROOT, "answers")
WORKING_DIR = os.environ.get("CERT_WORKING", "/kaggle/working")
ARTIFACTS_DIR = os.path.join(WORKING_DIR, "artifacts")

# ---------------------------------------------------------------------------
# Formats
# ---------------------------------------------------------------------------
DATE_FORMAT = "%m/%d/%Y %H:%M:%S"  # CERT r4.2 log timestamps (local time)

ID_RE = re.compile(r"^\{[A-Z0-9-]+\}$")
USER_RE = re.compile(r"^[A-Z]{3}\d{4}$")
PC_RE = re.compile(r"^PC-\d{4}$")

# ---------------------------------------------------------------------------
# Schemas: {file: [(column, type), ...]}  type in {str, int}
# ---------------------------------------------------------------------------
SCHEMAS = {
    "logon": [("id", str), ("date", str), ("user", str), ("pc", str), ("activity", str)],
    "device": [("id", str), ("date", str), ("user", str), ("pc", str), ("activity", str)],
    "file": [("id", str), ("date", str), ("user", str), ("pc", str), ("filename", str), ("content", str)],
    "http": [("id", str), ("date", str), ("user", str), ("pc", str), ("url", str), ("content", str)],
    "email": [("id", str), ("date", str), ("user", str), ("pc", str), ("to", str), ("cc", str),
              ("bcc", str), ("from", str), ("size", int), ("attachments", int), ("content", str)],
    "psychometric": [("employee_name", str), ("user_id", str), ("O", int), ("C", int),
                     ("E", int), ("A", int), ("N", int)],
    "insiders": [("dataset", float), ("scenario", int), ("details", str), ("user", str),
                 ("start", str), ("end", str)],
}

LOG_FILES = ["logon", "device", "file", "http", "email"]
LOG_PATHS = {name: os.path.join(R4_2_DIR, f"{name}.csv") for name in LOG_FILES}
LDAP_GLOB = os.path.join(R4_2_DIR, "LDAP", "*.csv")
PSYCHOMETRIC_PATH = os.path.join(R4_2_DIR, "psychometric.csv")
INSIDERS_PATH = os.path.join(ANSWERS_DIR, "insiders.csv")

# ---------------------------------------------------------------------------
# Allowed values
# ---------------------------------------------------------------------------
LOGON_ACTIVITIES = {"Logon", "Logoff"}
DEVICE_ACTIVITIES = {"Connect", "Disconnect"}

# ---------------------------------------------------------------------------
# Temporal split defaults (chronological, no shuffling)
# Leakage-safe default boundaries for CERT r4.2 (2010-01-02 .. 2011-05-17)
# ---------------------------------------------------------------------------
SPLIT_DEFAULTS = {
    "train_end": "2011-01-31",
    "calibration_end": "2011-03-31",
    # remainder after calibration_end is test
}

# ---------------------------------------------------------------------------
# Graph feature construction (Phase 5)
# ---------------------------------------------------------------------------
# LDAP monthly snapshots (2009-12 .. 2011-05): department(u) on day d comes
# from the LATEST snapshot STRICTLY BEFORE month(d) (published at month
# start; using it for month(d) itself would be same-month information).
LDAP_SNAPSHOT_FIRST_MONTH = "2009-12"

# "Other" label for file content that matches no declared magic-byte prefix.
FILE_TYPE_OTHER = "other"

# Rejected graph features (documented in the Phase 5 report; kept here so the
# rejection is machine-readable):
REJECTED_GRAPH_FEATURES = {
    "file_access_consistency_score": (
        "filename-based day-vs-past Jaccard: every r4.2 filename is globally "
        "unique (445,581 distinct in 445,581 rows, zero repeats even per user), "
        "so the score is identically 0. Replaced by file_type_consistency_score."
    ),
    "rare_file_access": (
        "filename-based rarity: same uniqueness property makes every access a "
        "'new file' (feature == file_access_count). Replaced by "
        "rare_file_type_access_count."
    ),
    "url_consistency_score": (
        "URL-based consistency would require scanning the 14.5 GB http.csv for "
        "an unproven signal; behavioral http_activity_count already exists. "
        "Deferred, not rejected on leakage grounds."
    ),
}

GRAPH_FEATURES = {
    "device_consistency_score": {
        "definition": ("Jaccard similarity between the set of PCs used by user u "
                       "on day d and the set of PCs u has used strictly before d."),
        "formula": ("|D(u,d) ∩ H_dev(u,d)| / |D(u,d) ∪ H_dev(u,d)|, "
                    "D(u,d) = {pc : logon by u on d}, "
                    "H_dev(u,d) = {pc : first_logon_day(u,pc) < d}; "
                    "0 if D or H empty."),
        "source_logs": ["logon.csv"],
        "historical_window": "all days strictly before d (first-use-day < d)",
        "baseline_population": "the user's own past (no cross-user statistics)",
        "prediction_time_availability": "end of day d (day-set) + strictly past logons",
        "normalization": "none (Jaccard is already [0,1])",
        "missing_unseen_behavior": ("no logons on d or no past devices -> 0.0; "
                                    "first use of a device on d itself -> not in past set"),
        "computational_cost": "O(events) via first-use-day table + day-set aggregates",
        "leakage_tests": "future-invariance, same-day-exclusion, cold start, duplicates",
    },
    "rare_device_usage_count": {
        "definition": ("Number of PCs used by user u on day d that u has never "
                       "used strictly before d (new-to-user devices)."),
        "formula": "|D(u,d) \\ H_dev(u,d)| with H_dev as above; first-ever use of a PC counts.",
        "source_logs": ["logon.csv"],
        "historical_window": "all days strictly before d",
        "baseline_population": "the user's own past",
        "prediction_time_availability": "end of day d",
        "normalization": "none (count)",
        "missing_unseen_behavior": "no logons on d -> 0",
        "computational_cost": "O(events)",
        "leakage_tests": "future-invariance, same-day-exclusion, cold start, duplicates",
    },
    "file_type_consistency_score": {
        "definition": ("Jaccard similarity between the set of file content types "
                       "accessed by u on day d and the set of types u has accessed "
                       "strictly before d. Types = declared magic-byte prefixes "
                       "(OLE2/PDF/ZIP) plus 'other'; filenames are globally unique "
                       "in r4.2, so types are the defensible file-access unit."),
        "formula": ("|T(u,d) ∩ H_type(u,d)| / |T(u,d) ∪ H_type(u,d)|, "
                    "T(u,d) = {type of file rows of u on d}, "
                    "H_type(u,d) = {type : first_file_type_day(u,type) < d}; "
                    "0 if T or H empty."),
        "source_logs": ["file.csv"],
        "historical_window": "all days strictly before d",
        "baseline_population": "the user's own past",
        "prediction_time_availability": "end of day d",
        "normalization": "none (Jaccard)",
        "missing_unseen_behavior": "no file rows on d or none before -> 0.0",
        "computational_cost": "O(events)",
        "leakage_tests": "future-invariance, cold start, empty history, unique-filename degeneracy",
    },
    "rare_file_type_access_count": {
        "definition": ("Number of file rows of user u on day d whose content type "
                       "u has never accessed strictly before d."),
        "formula": "count of rows r in file rows of u on d with type(r) ∉ H_type(u,d).",
        "source_logs": ["file.csv"],
        "historical_window": "all days strictly before d",
        "baseline_population": "the user's own past",
        "prediction_time_availability": "end of day d",
        "normalization": "none (count)",
        "missing_unseen_behavior": "no file rows on d -> 0",
        "computational_cost": "O(events)",
        "leakage_tests": "future-invariance, cold start, empty history",
    },
    "department_file_type_mismatch_count": {
        "definition": ("Number of file rows of user u on day d whose content type "
                       "was never accessed (strictly before d) by any user of u's "
                       "department. A context signal only; never an inferred label."),
        "formula": ("count of rows r in file rows of u on d with type(r) ∉ "
                    "H_dept_type(dept(u,d), d); "
                    "H_dept_type(δ,d) = {type : first_file_type_day(any user of δ, type) < d}; "
                    "dept(u,d) = department from the latest LDAP snapshot strictly "
                    "before month(d). 0 if u has no department or no file rows on d."),
        "source_logs": ["file.csv", "LDAP/*.csv"],
        "historical_window": "all days strictly before d (department-level)",
        "baseline_population": "the user's department (other users' strictly-past file rows)",
        "prediction_time_availability": "end of day d (strictly-past department history is observable)",
        "normalization": "none (count)",
        "missing_unseen_behavior": ("no department -> 0; department with no past file rows -> "
                                    "all day rows count (cold start, documented)"),
        "computational_cost": "O(events)",
        "leakage_tests": "future-invariance, department-month rule, cold-start department, unseen users",
    },
}

# ---------------------------------------------------------------------------
# Behavioral feature registry (Phase 7 + Phase 3 expansion)
# Each feature documents source log, definition and status.
# status: "defined" | "pending"
# ---------------------------------------------------------------------------
# sensitive_file_access_count: static a-priori rule over the file's own
# content signature (magic bytes), observable at event time. Declared from
# domain knowledge before any label inspection; NOT derived from the
# answer key (r4.2 filenames are random codes; scenario files define
# sensitivity per incident, which must not enter feature construction).
SENSITIVE_MAGIC_PREFIXES = [
    "D0-CF-11-E0-A1-B1-1A-E1",  # OLE2 (MS Office documents)
    "25-50-44-46-2D",           # "%PDF-" (PDF documents)
    "50-4B-03-04",              # "PK" (ZIP archives)
]

# unusual_access_count: window over strictly past days only (leakage-safe:
# never uses same-day or future logons). W=28 days declared a priori.
UNUSUAL_ACCESS_WINDOW_DAYS = 28

BEHAVIORAL_FEATURES = {
    "login_count": {
        "source": "logon.csv",
        "definition": "Number of rows with activity='Logon' per user-day.",
        "status": "defined",
    },
    "after_hours_login_count": {
        "source": "logon.csv",
        "definition": "Number of 'Logon' rows per user-day whose hour is < 8 or >= 18 "
                     "(8am-6pm workday baseline; CERT notes after-hours logons are significant).",
        "status": "defined",
    },
    "usb_connection_count": {
        "source": "device.csv",
        "definition": "Number of rows with activity='Connect' per user-day "
                     "(device events = removable media usage).",
        "status": "defined",
    },
    "file_access_count": {
        "source": "file.csv",
        "definition": "Number of file rows per user-day (file = copy to removable media).",
        "status": "defined",
    },
    "sensitive_file_access_count": {
        "source": "file.csv",
        "definition": "Number of file rows per user-day whose content starts with a "
                     "declared sensitive-document/archive magic-byte prefix "
                     f"({', '.join(SENSITIVE_MAGIC_PREFIXES)}: MS Office OLE2, PDF, "
                     "ZIP). Static a-priori rule; event-time information only "
                     "(never answer-key filenames).",
        "status": "defined",
    },
    "http_activity_count": {
        "source": "http.csv",
        "definition": "Number of http rows per user-day.",
        "status": "defined",
    },
    "unique_device_count": {
        "source": "logon.csv",
        "definition": "Number of distinct pc values in logon rows per user-day.",
        "status": "defined",
    },
    "unusual_access_count": {
        "source": "logon.csv",
        "definition": "Number of Logon events per user-day on a PC with no Logon by "
                     "the same user in the trailing "
                     f"{UNUSUAL_ACCESS_WINDOW_DAYS}-day window (day-W .. day-1; "
                     "strictly past, same-day excluded; first-ever use of a PC "
                     "counts). Window declared a priori; no future statistics.",
        "status": "defined",
    },
}
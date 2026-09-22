"""
migrate_csv_to_db.py — one-time import of the old CSV data into the database.

Usage:
    python migrate_csv_to_db.py                # imports local_*.csv from this folder
    python migrate_csv_to_db.py path/to/data   # e.g. the data/ folder of the old burnout-data GitHub repo

Imports into burnout.db by default, or into Postgres if DATABASE_URL is set
(env var or .streamlit/secrets.toml). Safe to re-run: rows already in the
database are skipped.
"""
import sys
from pathlib import Path
import pandas as pd
from sqlalchemy import Boolean, DateTime, Float, Integer, String, select, insert
import database as db

# table -> columns that identify a row, used to skip rows already imported
TABLES = {
    "students":          (db.students,          ["roll_number"]),
    "submissions":       (db.submissions,       ["roll_number", "timestamp"]),
    "replies":           (db.replies,           ["roll_number", "timestamp", "counselor_message"]),
    "counselor_actions": (db.counselor_actions, ["roll_number"]),
    "reminders":         (db.reminders,         ["created_at"]),
    "college_records":   (db.college_records,   ["roll_number", "branch", "section"]),
    "notifications":     (db.notifications,     ["id"]),
}

def convert(col, v):
    """CSV string -> the column's Python type (missing -> '' for text, None otherwise)."""
    if v is None or pd.isna(v) or (isinstance(v, str) and not v.strip()):
        return "" if isinstance(col.type, String) else None
    t = col.type
    if isinstance(t, Boolean):  return str(v).strip().lower() in ("true", "1", "yes")
    if isinstance(t, Integer):  return int(float(v))
    if isinstance(t, Float):    return float(v)
    if isinstance(t, DateTime): return pd.to_datetime(v).to_pydatetime()
    return str(v).strip()

def find_csv(folder: Path, key: str):
    for name in (f"local_{key}.csv", f"{key}.csv"):
        if (folder / name).exists():
            return folder / name
    return None

def main():
    folder = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent)
    engine = db.make_engine()
    print(f"Importing CSVs from {folder.resolve()}")
    print(f"Into {engine.url.render_as_string(hide_password=True)}\n")

    for key, (table, key_cols) in TABLES.items():
        path = find_csv(folder, key)
        if not path:
            print(f"  {key:18s} no CSV found, skipped")
            continue
        df = pd.read_csv(path, dtype=str)
        cols = [c for c in table.c if c.name in df.columns and c.autoincrement is not True]
        rows = [{c.name: convert(c, r[c.name]) for c in cols} for r in df.to_dict("records")]
        # fill required columns the old CSVs didn't have
        for r in rows:
            for c in table.c:
                if c.name not in r and not c.primary_key and not c.nullable and c.default is not None:
                    r[c.name] = c.default.arg

        with engine.begin() as conn:
            existing = {tuple(row) for row in conn.execute(select(*[table.c[k] for k in key_cols]))}
            new_rows, seen = [], set(existing)
            for r in rows:
                k = tuple(r.get(c) for c in key_cols)
                if k not in seen:
                    seen.add(k)
                    new_rows.append(r)
            if new_rows:
                conn.execute(insert(table), new_rows)
        print(f"  {key:18s} {len(new_rows)} imported, {len(rows) - len(new_rows)} already present")

    print("\nDone. Once you've checked the app, the old local_*.csv files can be archived.")

if __name__ == "__main__":
    main()

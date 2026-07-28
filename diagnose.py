#!/usr/bin/env python3
"""
Work out why the Streamlit app will not start.

Run it with the SAME interpreter you use for the app:

    python diagnose.py

and, if you launch the app some other way, also try:

    py -3.13 diagnose.py

It checks the interpreter, the package versions, every parquet the app reads,
and the specific columns app.py expects — then prints a short verdict. Paste
the whole output.
"""

import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

problems = []

print("=" * 62)
print("INTERPRETER")
print("=" * 62)
print(f"executable : {sys.executable}")
print(f"version    : {sys.version.split()[0]}")

print()
print("=" * 62)
print("PACKAGES")
print("=" * 62)
import importlib.metadata as md
for pkg, pinned in [("streamlit", "1.60.0"), ("pandas", "3.0.5"), ("numpy", "2.5.1"),
                    ("matplotlib", "3.11.1"), ("pyarrow", "24.0.0"),
                    ("scipy", "1.18.0"), ("requests", "2.34.2")]:
    try:
        got = md.version(pkg)
        flag = "" if got == pinned else f"  <-- requirements.txt pins {pinned}"
        print(f"{pkg:<12} {got}{flag}")
        if got != pinned:
            problems.append(f"{pkg} is {got}, not the pinned {pinned}")
    except Exception:
        print(f"{pkg:<12} NOT INSTALLED  <-- required")
        problems.append(f"{pkg} is not installed for this interpreter")

print()
print("=" * 62)
print("PARQUET FILES")
print("=" * 62)
try:
    import pandas as pd
except Exception:
    print("pandas will not import — nothing else can be checked")
    traceback.print_exc()
    sys.exit(1)

# file -> columns app.py relies on
EXPECTED = {
    "practices.parquet": ["PracNo", "PracticeName", "LCG", "RegisteredPatients",
                          "DepQuintile"],
    "prescribing.parquet": ["Practice", "VTM_NM", "TotalItems", "ActualCost",
                            "TotalQuantity", "Year", "Month", "RegisteredPatients"],
    "prescribing_practice_monthly.parquet": ["practice", "year", "month",
                                             "bnf_chapter", "total_items"],
    "prescribing_lcg_monthly.parquet": ["lcg", "year", "month", "bnf_chapter"],
    "standardised_rates_practice.parquet": ["practice", "year", "month",
                                            "bnf_chapter", "starpu"],
    "standardised_rates_lcg.parquet": ["lcg", "year", "month", "bnf_chapter",
                                       "starpu"],
    "therapeutic_area_ni_monthly.parquet": ["year", "month", "therapeutic_area",
                                            "total_items"],
    "therapeutic_area_practice_monthly.parquet": ["practice", "year", "month",
                                                  "therapeutic_area"],
    "starpu_denominators_practice.parquet": ["year", "practice", "bnf_chapter",
                                             "starpu", "total_population"],
    "starpu_denominators_lcg.parquet": ["year", "lcg", "bnf_chapter", "starpu"],
    "qof.parquet": ["Practice", "QOF_Domain", "QOF_Indicator"],
    "prevalence.parquet": ["Practice", "QOF_Domain", "Prevalence"],
}

for fname, needed in EXPECTED.items():
    path = os.path.join(DATA, fname)
    if not os.path.exists(path):
        print(f"{fname:<44} MISSING")
        problems.append(f"{fname} is missing from data/")
        continue
    try:
        df = pd.read_parquet(path)
        missing = [c for c in needed if c not in df.columns]
        size_mb = os.path.getsize(path) / 1e6
        status = "ok" if not missing else f"MISSING COLUMNS {missing}"
        print(f"{fname:<44} {len(df):>10,} rows  {size_mb:>7.1f}MB  {status}")
        if missing:
            problems.append(f"{fname} has no {missing} column(s)")
            print(f"    actual columns: {list(df.columns)}")
    except Exception as exc:
        print(f"{fname:<44} FAILED TO READ: {type(exc).__name__}: {exc}")
        problems.append(f"{fname} will not load: {exc}")

print()
print("=" * 62)
print("IMPORTING app.py")
print("=" * 62)
# Import without running Streamlit, to surface syntax errors and import-time faults.
try:
    import ast
    with open(os.path.join(HERE, "app.py"), encoding="utf-8") as fh:
        src = fh.read()
    ast.parse(src)
    print("app.py parses cleanly")
    print(f"app.py is {len(src.encode('utf-8')):,} bytes, "
          f"{src.count(chr(10)):,} lines")
except SyntaxError as exc:
    print(f"SYNTAX ERROR in app.py at line {exc.lineno}: {exc.msg}")
    print(f"  {exc.text}")
    problems.append(f"app.py has a syntax error at line {exc.lineno}")
except Exception as exc:
    print(f"could not read app.py: {exc}")
    problems.append(f"app.py unreadable: {exc}")

print()
print("=" * 62)
print("VERDICT")
print("=" * 62)
if problems:
    for p in problems:
        print(f"  - {p}")
else:
    print("  Everything the app reads looks fine from here.")
    print("  The failure is happening at runtime — please copy the red error")
    print("  box from the browser, or the traceback from the terminal.")

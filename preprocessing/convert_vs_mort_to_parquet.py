"""
Convert NCHS Mortality Multiple Cause-of-Death public-use fixed-width file
(e.g. VS24MORT.DUSMCPUB_r20251208) to a parquet matching the schema that
process_mmcd.py consumes.

Output schema:
    sex                       : 'M' / 'F'
    age_lower_bound           : float, age expressed in milliseconds
                                (process_mmcd.py divides by 1000/60/60/24/365.25)
    record_axis_conditions    : list[str], ICD-10 codes from the record-axis section

Record layout (2024 file, 1-indexed positions; see
https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Dataset_Documentation/DVS/mortality/2024-Mortality-Public-Use-File-Documentation.pdf):
    69         Sex ('M' / 'F')
    70         Age class (1=years, 2=months, 4=days, 5=hours, 6=minutes, 9=not stated)
    71-73      Age value (3 digits; 999 = not stated within bin)
    341-342    Number of record-axis conditions (00-20)
    344-443    Record-axis conditions: 20 slots x 5 chars; positions 1-4 = ICD-10 code

Usage:
    python preprocessing/convert_vs_mort_to_parquet.py \\
        /Users/alexepstein/Downloads/VS24MORT.DUSMCPUB_r20251208 \\
        raw_data/MMCD_2024.parquet
"""

import argparse
import os
import sys
import time

import pandas as pd

MS_PER_YEAR = 365.25 * 24 * 60 * 60 * 1000
CLS_TO_MS = {
    '1': MS_PER_YEAR,
    '2': MS_PER_YEAR / 12,
    '4': 24 * 60 * 60 * 1000,
    '5': 60 * 60 * 1000,
    '6': 60 * 1000,
}

CHUNK_ROWS = 250_000


def parse_age(line):
    cls = line[69]
    factor = CLS_TO_MS.get(cls)
    if factor is None:
        return None
    val_str = line[70:73]
    try:
        val = int(val_str)
    except ValueError:
        return None
    if val == 999:
        return None
    return val * factor


def parse_record_axis(line):
    block = line[343:443]
    codes = []
    for i in range(20):
        code = block[i * 5:i * 5 + 4].strip()
        if code:
            codes.append(code)
    return codes


def parse_chunk(lines):
    sexes, ages, conds = [], [], []
    for line in lines:
        if len(line) < 443:
            continue
        sex = line[68]
        if sex not in ('M', 'F'):
            continue
        sexes.append(sex)
        ages.append(parse_age(line))
        conds.append(parse_record_axis(line))
    return pd.DataFrame({
        'sex': pd.array(sexes, dtype='string'),
        'age_lower_bound': ages,
        'record_axis_conditions': conds,
    })


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('input_path')
    ap.add_argument('output_path')
    ap.add_argument('--chunk-rows', type=int, default=CHUNK_ROWS)
    args = ap.parse_args()

    out_dir = os.path.dirname(os.path.abspath(args.output_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    if os.path.exists(args.output_path):
        os.remove(args.output_path)

    import fastparquet

    total_in = 0
    total_out = 0
    t0 = time.time()
    first = True
    chunk = []

    with open(args.input_path, 'r') as f:
        for line in f:
            chunk.append(line.rstrip('\n'))
            if len(chunk) >= args.chunk_rows:
                df = parse_chunk(chunk)
                fastparquet.write(
                    args.output_path,
                    df,
                    append=not first,
                    compression='SNAPPY',
                    object_encoding={
                        'sex': 'utf8',
                        'record_axis_conditions': 'json',
                    },
                )
                first = False
                total_in += len(chunk)
                total_out += len(df)
                chunk = []
                elapsed = time.time() - t0
                rate = total_in / elapsed if elapsed > 0 else 0
                print(f"  {total_in:>10,} rows read  /  {total_out:>10,} written  "
                      f"({rate:>7,.0f} rows/sec)", flush=True)

    if chunk:
        df = parse_chunk(chunk)
        if len(df):
            fastparquet.write(
                args.output_path,
                df,
                append=not first,
                compression='SNAPPY',
                object_encoding={
                    'sex': 'utf8',
                    'record_axis_conditions': 'json',
                },
            )
        total_in += len(chunk)
        total_out += len(df)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s")
    print(f"Read   : {total_in:,} rows")
    print(f"Written: {total_out:,} rows")
    size = os.path.getsize(args.output_path)
    print(f"Output : {args.output_path} ({size / 1024 / 1024:.1f} MiB)")


if __name__ == '__main__':
    sys.exit(main())

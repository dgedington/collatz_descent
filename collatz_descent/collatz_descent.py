#!/usr/bin/env python3
"""
Collatz Direct Descent Simulator with δ Analysis
=================================================
Simulates all residue classes N ≡ b (mod 2^m) and computes the δ quantity
that determines whether each class descends, ascends, or loops.

Master invariant:   2^m · M  =  3^n · N  +  f
δ(N)            =  f/N − gap          where gap = 2^m − 3^n
  δ < 0  →  M < N  (descent)
  δ = 0  →  M = N  (loop: N = f/gap)
  δ > 0  →  M > N  (ascent)

Output modes
  default (m < 10)  : full per-residue-class table to stdout
  default (m ≥ 10)  : Pascal-cell aggregation to stdout
  --pascal          : Pascal-cell aggregation (any m)
  --full            : full per-residue-class table (any m)
  --output FILE     : write the selected table to FILE (.csv or .tsv supported)
  --filter FILTER   : restrict rows (see --help for values)

Usage examples
  python collatz_descent.py --m 6
  python collatz_descent.py --m 6  --full --output results.csv
  python collatz_descent.py --m 20 --pascal
  python collatz_descent.py --m 12 --filter loops
  python collatz_descent.py --m 8  --filter ascent --output ascent.tsv
  python collatz_descent.py --m 14 --filter groupC --pascal
"""

import argparse
import csv
import sys
from fractions import Fraction
from math import comb

# ── NumPy (optional fast path) ────────────────────────────────────
try:
    import numpy as np
    NUMPY = True
except ImportError:
    NUMPY = False


# ═══════════════════════════════════════════════════════════════════
#  SIMULATION ENGINE
# ═══════════════════════════════════════════════════════════════════

def simulate_numpy(m):
    """
    Vectorised simulation using NumPy.
    Runs all 2^m residue classes in parallel through m steps.

    int64 is safe for m ≤ 40 (f_max stays well under 2^32 in practice).
    For m > 40 we fall back to Python object arrays (arbitrary precision).

    Returns
    -------
    b_arr : array of final residues  (shape 2^m)
    n_arr : array of odd-step counts (shape 2^m)
    f_arr : array of f values        (shape 2^m)
    """
    size  = 2 ** m
    dtype = np.int64 if m <= 40 else object

    b_arr = np.arange(size, dtype=dtype)
    f_arr = np.zeros(size, dtype=dtype)
    n_arr = np.zeros(size, dtype=dtype)

    for step in range(1, m + 1):
        is_odd = (b_arr % 2 == 1)
        pow2   = dtype(2 ** (step - 1))

        # Odd step
        f_arr[is_odd]  = 3 * f_arr[is_odd] + pow2
        b_arr[is_odd]  = (3 * b_arr[is_odd] + 1) // 2
        n_arr[is_odd] += 1

        # Even step
        b_arr[~is_odd] = b_arr[~is_odd] // 2

    return b_arr, n_arr, f_arr


def simulate_python(m):
    """
    Pure-Python fallback when NumPy is unavailable.
    Returns (b_list, n_list, f_list) as plain Python lists.
    """
    b_list, n_list, f_list = [], [], []
    for b_init in range(2 ** m):
        b_cur = b_init
        f = n = 0
        for step in range(1, m + 1):
            if b_cur % 2 == 1:
                f     = 3 * f + 2 ** (step - 1)
                b_cur = (3 * b_cur + 1) // 2
                n    += 1
            else:
                b_cur //= 2
        b_list.append(b_cur)
        n_list.append(n)
        f_list.append(f)
    return b_list, n_list, f_list


def run_simulation(m):
    """Dispatch to NumPy or Python engine; always return plain Python lists."""
    if NUMPY:
        b_arr, n_arr, f_arr = simulate_numpy(m)
        return list(map(int, b_arr)), list(map(int, n_arr)), list(map(int, f_arr))
    return simulate_python(m)


# ═══════════════════════════════════════════════════════════════════
#  RESULT BUILDING
# ═══════════════════════════════════════════════════════════════════

def build_rows(m, b_list, n_list, f_list):
    """
    Convert raw simulation output into result dicts, one per residue class.
    δ sign uses integer arithmetic; Fraction is computed only for display.
    """
    rows = []
    for idx, (b, n, f) in enumerate(zip(b_list, n_list, f_list)):
        gap   = 2**m - 3**n
        group = 'A' if gap > 0 else 'C'

        # δ sign via integer comparison — avoids division entirely
        if b == 0:
            delta_sign = -1          # all-even: δ = −gap < 0
        else:
            diff       = f - gap * b # same sign as δ(b) × b
            delta_sign = (1 if diff > 0 else (-1 if diff < 0 else 0))

        # Loop detection: δ = 0 iff f == gap * b  (and gap > 0, b > 0)
        loop_N = None
        if delta_sign == 0 and gap > 0 and b > 0:
            loop_N = b   # smallest valid N for this path that loops

        # Exact δ as Fraction — only materialised for display
        if b == 0:
            delta_frac = Fraction(-gap)
        else:
            delta_frac = Fraction(f, b) - gap

        rows.append({
            'b':          idx,
            'n':          n,
            'f':          f,
            'b_res':      b,
            'gap':        gap,
            'group':      group,
            'delta_sign': delta_sign,
            'delta_frac': delta_frac,
            'loop_N':     loop_N,
        })
    return rows


def build_pascal_cells(m, rows):
    """Aggregate rows by Pascal cell (m, n). Returns one dict per n value."""
    from collections import defaultdict
    buckets = defaultdict(lambda: {'neg': 0, 'zero': 0, 'pos': 0, 'loops': []})

    for r in rows:
        bk = buckets[r['n']]
        s  = r['delta_sign']
        if   s < 0: bk['neg']  += 1
        elif s > 0: bk['pos']  += 1
        else:
            bk['zero'] += 1
            if r['loop_N'] is not None:
                bk['loops'].append(r['loop_N'])

    cells = []
    for n in range(m + 1):
        bk  = buckets[n]
        gap = 2**m - 3**n
        cells.append({
            'n':        n,
            'binomial': comb(m, n),
            'gap':      gap,
            'group':    'A' if gap > 0 else 'C',
            'neg':      bk['neg'],
            'zero':     bk['zero'],
            'pos':      bk['pos'],
            'loops':    sorted(set(bk['loops'])),
        })
    return cells


# ═══════════════════════════════════════════════════════════════════
#  FILTERING
# ═══════════════════════════════════════════════════════════════════

FILTER_HELP = """
  all      : all residue classes (default)
  descent  : only δ < 0 classes
  ascent   : only δ > 0 classes
  loops    : only δ = 0 classes (loop candidates)
  groupA   : only Group A cells (gap > 0)
  groupC   : only Group C cells (gap < 0)
  newloops : only loops with N > 2 (potential new cycles)
"""

def apply_row_filter(rows, fname):
    fname = fname.lower()
    if fname == 'all':      return rows
    if fname == 'descent':  return [r for r in rows if r['delta_sign'] < 0]
    if fname == 'ascent':   return [r for r in rows if r['delta_sign'] > 0]
    if fname == 'loops':    return [r for r in rows if r['delta_sign'] == 0]
    if fname == 'groupa':   return [r for r in rows if r['group'] == 'A']
    if fname == 'groupc':   return [r for r in rows if r['group'] == 'C']
    if fname == 'newloops': return [r for r in rows if r['loop_N'] is not None and r['loop_N'] > 2]
    raise ValueError(f"Unknown filter '{fname}'.{FILTER_HELP}")

def apply_cell_filter(cells, fname):
    fname = fname.lower()
    if fname == 'all':      return cells
    if fname == 'descent':  return [c for c in cells if c['neg']  > 0]
    if fname == 'ascent':   return [c for c in cells if c['pos']  > 0]
    if fname == 'loops':    return [c for c in cells if c['zero'] > 0]
    if fname == 'groupa':   return [c for c in cells if c['group'] == 'A']
    if fname == 'groupc':   return [c for c in cells if c['group'] == 'C']
    if fname == 'newloops': return [c for c in cells if any(lN > 2 for lN in c['loops'])]
    raise ValueError(f"Unknown filter '{fname}'.{FILTER_HELP}")


# ═══════════════════════════════════════════════════════════════════
#  PATH RECONSTRUCTION  (only used for full-table display with m ≤ 20)
# ═══════════════════════════════════════════════════════════════════

def rebuild_paths(m):
    """Reconstruct the parity vector string for every b in 0..2^m-1."""
    paths = []
    for b_init in range(2 ** m):
        b_cur, path = b_init, []
        for _ in range(m):
            if b_cur % 2 == 1:
                path.append('O')
                b_cur = (3 * b_cur + 1) // 2
            else:
                path.append('E')
                b_cur //= 2
        paths.append(''.join(path))
    return paths


# ═══════════════════════════════════════════════════════════════════
#  FORMATTING
# ═══════════════════════════════════════════════════════════════════

def _loop_tag(loop_N):
    return f"N={loop_N}(attr)" if loop_N <= 2 else f"N={loop_N} *** NEW LOOP ***"


def _delta_str(frac):
    return f"{int(frac):>13}" if frac.denominator == 1 else f"{float(frac):>13.4f}"


def format_full_table(m, rows, paths=None, writer=None):
    """
    Render the full per-residue-class table.
    paths: list of path strings (None if m > 20 — paths are omitted).
    writer: csv.writer for file output, or None for stdout.
    """
    pw = m  # path column width

    if writer:
        header = (['b', 'path', 'n', 'f', 'gap', 'group',
                   'delta_num', 'delta_den', 'delta_float', 'loop_N']
                  if paths else
                  ['b', 'n', 'f', 'gap', 'group',
                   'delta_num', 'delta_den', 'delta_float', 'loop_N'])
        writer.writerow(header)
    else:
        print(f"\n{'═'*82}")
        print(f"  m = {m}   full table   ({len(rows)} rows shown)")
        print(f"{'═'*82}")
        if paths:
            print(f"  {'b':>4} │ {'Path':<{pw}} │ {'n':>2} │ {'f':>9} │ "
                  f"{'gap':>10} │ Grp │ {'δ(b)':>13} │ Loop")
            print(f"  {'─'*4}┼{'─'*(pw+2)}┼{'─'*4}┼{'─'*11}┼"
                  f"{'─'*12}┼{'─'*5}┼{'─'*15}┼{'─'*22}")
        else:
            print(f"  {'b':>6} │ {'n':>2} │ {'f':>9} │ "
                  f"{'gap':>10} │ Grp │ {'δ(b)':>13} │ Loop")
            print(f"  {'─'*6}┼{'─'*4}┼{'─'*11}┼"
                  f"{'─'*12}┼{'─'*5}┼{'─'*15}┼{'─'*22}")

    for r in rows:
        d        = r['delta_frac']
        d_str    = _delta_str(d)
        loop_str = _loop_tag(r['loop_N']) if r['loop_N'] is not None else ''
        path_str = paths[r['b']] if paths else None

        if writer:
            row_data = (
                [r['b'], path_str, r['n'], r['f'], r['gap'], r['group'],
                 d.numerator, d.denominator, float(d),
                 r['loop_N'] if r['loop_N'] is not None else '']
                if paths else
                [r['b'], r['n'], r['f'], r['gap'], r['group'],
                 d.numerator, d.denominator, float(d),
                 r['loop_N'] if r['loop_N'] is not None else '']
            )
            writer.writerow(row_data)
        else:
            if paths:
                print(f"  {r['b']:>4} │ {path_str:<{pw}} │ {r['n']:>2} │ "
                      f"{r['f']:>9} │ {r['gap']:>10} │ {r['group']:>3} │ "
                      f"{d_str} │ {loop_str}")
            else:
                print(f"  {r['b']:>6} │ {r['n']:>2} │ "
                      f"{r['f']:>9} │ {r['gap']:>10} │ {r['group']:>3} │ "
                      f"{d_str} │ {loop_str}")


def format_pascal_table(m, cells, writer=None):
    """Render the Pascal-cell aggregation table."""
    if writer:
        writer.writerow(['m', 'n', 'C(m,n)', 'gap', 'group',
                         'delta_neg', 'delta_zero', 'delta_pos', 'loop_N_values'])
    else:
        print(f"\n{'═'*80}")
        print(f"  m = {m}   Pascal-cell aggregation   "
              f"({2**m} residue classes → {len(cells)} cells shown)")
        print(f"{'═'*80}")
        print(f"  {'n':>3} │ {'C(m,n)':>8} │ {'gap':>11} │ Grp │ "
              f"{'δ<0':>7} │ {'δ=0':>6} │ {'δ>0':>7} │ loops")
        print(f"  {'─'*3}┼{'─'*10}┼{'─'*13}┼{'─'*5}┼"
              f"{'─'*9}┼{'─'*8}┼{'─'*9}┼{'─'*24}")

    for c in cells:
        loop_str = ', '.join(_loop_tag(lN) for lN in c['loops'])
        if writer:
            writer.writerow([m, c['n'], c['binomial'], c['gap'], c['group'],
                             c['neg'], c['zero'], c['pos'],
                             ';'.join(str(lN) for lN in c['loops'])])
        else:
            print(f"  {c['n']:>3} │ {c['binomial']:>8} │ {c['gap']:>11} │ "
                  f"{c['group']:>3} │ {c['neg']:>7} │ {c['zero']:>6} │ "
                  f"{c['pos']:>7} │ {loop_str}")


def print_summary(m, rows):
    """Print the summary block."""
    total     = 2 ** m
    n_neg     = sum(r['delta_sign'] < 0 for r in rows)
    n_zero    = sum(r['delta_sign'] == 0 for r in rows)
    n_pos     = sum(r['delta_sign'] > 0 for r in rows)
    new_loops = [r for r in rows if r['loop_N'] is not None and r['loop_N'] > 2]

    print(f"\n{'─'*60}")
    print(f"  Summary   m={m}   ({total:,} residue classes total)")
    print(f"{'─'*60}")
    print(f"  Engine        : {'NumPy' if NUMPY else 'Pure Python'}")
    print(f"  δ < 0 descent : {n_neg:>8,}   ({100*n_neg/total:5.1f}%)")
    print(f"  δ = 0 loop    : {n_zero:>8,}")
    for r in rows:
        if r['delta_sign'] == 0:
            tag = '(attractor)' if r['loop_N'] <= 2 else '*** NEW LOOP ***'
            print(f"                   n={r['n']}  loop_N={r['loop_N']}  {tag}")
    print(f"  δ > 0 ascent  : {n_pos:>8,}   ({100*n_pos/total:5.1f}%)")

    if new_loops:
        print(f"\n  *** {len(new_loops)} NEW LOOP(S) — COLLATZ CONJECTURE POTENTIALLY VIOLATED ***")
        for r in new_loops:
            print(f"      n={r['n']}  loop_N={r['loop_N']}  f={r['f']}  gap={r['gap']}")
    else:
        print(f"\n  No new loops found (only known trivial cycle at N≤2).")


# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        prog='collatz_descent.py',
        description='Collatz Descent Simulator with δ Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
output modes
  default          : full table for m < 10, Pascal aggregation for m ≥ 10
  --pascal         : Pascal-cell aggregation (always, any m)
  --full           : full per-residue-class table (always, any m)

filter values (--filter){FILTER_HELP}
examples
  python collatz_descent.py --m 6
  python collatz_descent.py --m 6  --full --output results.csv
  python collatz_descent.py --m 20 --pascal
  python collatz_descent.py --m 12 --filter loops
  python collatz_descent.py --m 8  --filter ascent --output ascent.tsv
  python collatz_descent.py --m 14 --filter groupC --pascal
        """
    )

    parser.add_argument('--m', type=int, required=True,
                        help='Row depth. Analyses all 2^m residue classes.')
    parser.add_argument('--pascal', action='store_true',
                        help='Pascal-cell aggregation output (m+1 rows).')
    parser.add_argument('--full', action='store_true',
                        help='Full per-residue-class table (2^m rows).')
    parser.add_argument('--output', metavar='FILE',
                        help='Write table to FILE (.csv or .tsv).')
    parser.add_argument('--filter', metavar='FILTER', default='all',
                        help='Filter rows: all descent ascent loops '
                             'groupA groupC newloops  (default: all)')
    parser.add_argument('--no-summary', action='store_true',
                        help='Suppress the summary block.')

    args = parser.parse_args()

    # ── Validation ────────────────────────────────────────────────
    if args.m < 1:
        print("Error: m must be ≥ 1.", file=sys.stderr); sys.exit(1)

    if args.pascal and args.full:
        print("Error: --pascal and --full are mutually exclusive.", file=sys.stderr)
        sys.exit(1)

    if args.m > 28 and args.full:
        try:
            resp = input(f"Warning: --full at m={args.m} → {2**args.m:,} rows. Continue? (y/n): ")
        except EOFError:
            resp = 'n'
        if resp.strip().lower() != 'y':
            sys.exit(0)

    if args.m > 30:
        try:
            resp = input(f"Warning: m={args.m} → {2**args.m:,} residue classes. Continue? (y/n): ")
        except EOFError:
            resp = 'n'
        if resp.strip().lower() != 'y':
            sys.exit(0)

    # ── Output mode ───────────────────────────────────────────────
    use_pascal = args.pascal or (not args.full and args.m >= 10)
    use_full   = not use_pascal

    # ── Simulate ──────────────────────────────────────────────────
    b_list, n_list, f_list = run_simulation(args.m)
    rows  = build_rows(args.m, b_list, n_list, f_list)
    cells = build_pascal_cells(args.m, rows)

    # ── Filter ────────────────────────────────────────────────────
    try:
        f_rows  = apply_row_filter(rows,  args.filter)
        f_cells = apply_cell_filter(cells, args.filter)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr); sys.exit(1)

    # ── Open output file ──────────────────────────────────────────
    out_file   = None
    csv_writer = None
    if args.output:
        delim    = '\t' if args.output.endswith('.tsv') else ','
        out_file  = open(args.output, 'w', newline='', encoding='utf-8')
        csv_writer = csv.writer(out_file, delimiter=delim)

    # ── Render ────────────────────────────────────────────────────
    try:
        if use_full:
            paths = rebuild_paths(args.m) if args.m <= 20 else None
            format_full_table(args.m, f_rows, paths, csv_writer)
        else:
            format_pascal_table(args.m, f_cells, csv_writer)
    finally:
        if out_file:
            out_file.close()

    if args.output:
        print(f"\n  Output written to: {args.output}")

    if not args.no_summary:
        print_summary(args.m, rows)


if __name__ == '__main__':
    main()
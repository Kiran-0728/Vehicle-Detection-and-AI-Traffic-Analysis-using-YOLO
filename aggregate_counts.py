"""
Aggregate per-frame counts CSV into fixed interval counts (e.g., 15 minutes).
Run: python aggregate_counts.py --raw data/counts_raw_perframe.csv --out data/counts_15min.csv --freq 15T
"""
import pandas as pd, argparse, os
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--freq', default='15T')
    args = parser.parse_args()
    df = pd.read_csv(args.raw, parse_dates=['timestamp'])
    df = df.set_index('timestamp').resample(args.freq).sum().reset_index()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    print("Wrote aggregated counts to", args.out)

if __name__ == '__main__':
    main()

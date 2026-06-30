#!/usr/bin/env python3
import argparse
import yaml
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/data_config.yaml')
    parser.add_argument('--output_dir', type=str, default='data/raw')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


if __name__ == '__main__':
    main()

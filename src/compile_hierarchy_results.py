import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from loguru import logger
import argparse


parser = argparse.ArgumentParser(description='Taskonomy evaluation script')
parser.add_argument('--results_dir', default="../results_imagenet_novel", type=str,
                    help='Directory where results were saved, and those that will be compiled')
parser.add_argument('--compiled_results_dir', default="../compiled_results", type=str,
                    help='Directory where compiled results will be saved')

if __name__=="__main__":
    args = parser.parse_args()
    logger.debug(f"args: {args}")

    if not os.path.exists(args.compiled_results_dir):
        os.makedirs(args.compiled_results_dir, exist_ok=True)

    metrics = None
    for fname in os.listdir(args.results_dir):
        fp = os.path.join(args.results_dir, fname)
        df = pd.read_csv(fp)[-1:]
        if metrics is None:
            metrics = df.copy()
        else:
            metrics = pd.concat([metrics, df], ignore_index=True)

    metrics = metrics.drop("Unnamed: 0", axis=1)
    metrics_fp = os.path.join(args.compiled_results_dir, f"{args.results_dir.split('/')[-1]}_hierarchy.csv")
    metrics.to_csv(metrics_fp)
import os
import argparse
import yaml
import shutil
import sys
import logging


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file",
        default="./configs/mvsec_indoor_no_timeaware.yaml",
        help="Config file yaml path",
        type=str,
    )
    parser.add_argument(
        "--eval",
        help="Add for evaluation run",
        action="store_true",
    )
    parser.add_argument(
        "--log", help="Log level: [debug, info, warning, error, critical]", type=str, default="info"
    )
    args = parser.parse_args()
    with open(args.config_file, "r") as f:
        config = yaml.safe_load(f)
    return config, args


def save_config(save_dir: str, file_name: str):
    """Save configuration"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    shutil.copy(file_name, save_dir)

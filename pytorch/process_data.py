import os
from pathlib import Path
import shutil

import numpy as np
import pandas as pd


def create_label_folders(df: pd.DataFrame, img_p: Path, split: str):
    labels = df.columns[1:]
    df[labels] = df[labels].astype(int)
    print("Number of labels:", len(labels))
    for col in labels:
        part_df = df[df[col] == 1]
        dst_p = img_p.parent / split / col
        dst_p.mkdir(exist_ok=True, parents=True)
        for image in part_df["image"].to_list():
            shutil.copy(src=img_p / (image + ".jpg"), dst=dst_p / (image + ".jpg"))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--img_p", required=True, type=str, help="Path to images for split")
    parser.add_argument("--csv_p", required=True, type=str, help="Path to label .csv file")
    parser.add_argument(
        "--split", required=True, choices=["train", "val", "test"], help="Split for creating subdirectory"
    )
    args = parser.parse_args()

    # csv_p = Path("/home/kti03/Data/ISIC2018/labels.csv")
    # img_p = Path("/home/kti03/Data/ISIC2018/images")
    img_p = Path(args.img_p)
    assert img_p.is_dir()
    csv_p = Path(args.csv_p)
    assert csv_p.is_file()
    df = pd.read_csv(csv_p)

    create_label_folders(df, img_p, split=args.split)

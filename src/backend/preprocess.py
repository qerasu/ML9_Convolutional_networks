import glob
import os
import pandas as pd
from sklearn.model_selection import train_test_split


class Preprocessor:
    def __init__(self, csv_path, images_dir, val_fraction=0.33, random_state=42):
        self.csv_path = csv_path
        self.images_dir = images_dir
        self.val_fraction = val_fraction
        self.random_state = random_state
        self._path_cache = None


    def _stem_to_path(self):
        if self._path_cache is not None:
            return self._path_cache

        mapping = {}
        for path in glob.glob(os.path.join(self.images_dir, "*.jpg")):
            stem = os.path.splitext(os.path.basename(path))[0]
            mapping[stem] = path

        self._path_cache = mapping

        return mapping


    def _attach_paths(self, df):
        df = df.copy()
        df["filepath"] = df["img_IDS"].map(self._stem_to_path())

        return df


    def _load_and_dedup_ids(self):
        raw_df = pd.read_csv(self.csv_path)

        before = len(raw_df)
        dedup_df = raw_df.drop_duplicates(subset=["img_IDS"], keep="first").reset_index(drop=True)
        print(f"Removed {before - len(dedup_df)} duplicate rows by img_IDS.")

        return raw_df, dedup_df


    def _clean(self, dedup_df):
        df = self._attach_paths(dedup_df)

        missing = int(df["filepath"].isna().sum())
        if missing:
            print(f"[WARNING] {missing} row(s) have no matching image files — dropped.")

        df = df.dropna(subset=["filepath"])
        clean_df = df.drop(columns=["filepath"])

        return clean_df


    def _build_report(self, raw_df, dedup_df, clean_df):
        to_dict = lambda s: {str(k): int(v) for k, v in s.to_dict().items()}

        report = {
            "rows_raw": len(raw_df),
            "removed_duplicate_ids": len(raw_df) - len(dedup_df),
            "rows_after_dedup": len(dedup_df),
            "rows_with_files": len(clean_df),
            "rows_missing_files": len(dedup_df) - len(clean_df),
            "label_distribution_raw": to_dict(raw_df["Label"].value_counts()),
            "label_distribution_clean": to_dict(clean_df["Label"].value_counts()),
        }

        return report


    def split(self, return_report=False):
        raw_df, dedup_df = self._load_and_dedup_ids()
        clean_df = self._clean(dedup_df)

        train_df, val_df = train_test_split(
            clean_df,
            test_size=self.val_fraction,
            random_state=self.random_state,
            stratify=clean_df["Label"],
        )
        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)

        if return_report:
            report = self._build_report(raw_df, dedup_df, clean_df)
            return train_df, val_df, report

        return train_df, val_df
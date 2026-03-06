import hashlib
import glob
import os
import pandas as pd
from sklearn.model_selection import train_test_split


CSV_PATH = "../datasets/Train.csv"
IMAGES_DIR = "../datasets/Images"


class Preprocessor:
    def __init__(self, csv_path=CSV_PATH, images_dir=IMAGES_DIR, val_fraction=0.33, random_state=42):
        self.csv_path = csv_path
        self.images_dir = images_dir
        self.val_fraction = val_fraction
        self.random_state = random_state
        self._path_cache = None


    @staticmethod
    def _file_hash(path):
        digest = hashlib.blake2b(digest_size=16)
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(8192), b""):
                digest.update(chunk)

        return digest.hexdigest()


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


    def _find_mislabeled_groups(self, df):
        conflicts = df.groupby("img_hash")["Label"].nunique()
        conflicts = conflicts[conflicts > 1].index

        groups = []
        for h in conflicts:
            rows = df.loc[df["img_hash"] == h, ["img_IDS", "Label"]]
            groups.append({
                "hash": h,
                "samples": [
                    {"img_IDS": str(r["img_IDS"]), "Label": int(r["Label"])}
                    for r in rows.to_dict(orient="records")
                ],
            })

        return groups


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

        df = df.dropna(subset=["filepath"]).copy()

        if df.empty:
            return df, df

        df["img_hash"] = df["filepath"].apply(self._file_hash)
        with_paths_df = df.copy()

        mislabeled = self._find_mislabeled_groups(df)
        if mislabeled:
            print(f"[MISLABEL] Found {len(mislabeled)} potential mislabeled groups.")
        else:
            print("[MISLABEL] No conflicting labels found.")

        before = len(df)
        df = df.drop_duplicates(subset=["img_hash"], keep="first").reset_index(drop=True)
        print(f"[DEDUP] Removed {before - len(df)} pixel-identical duplicates.")

        clean_df = df.drop(columns=["filepath", "img_hash"])

        return with_paths_df, clean_df


    def _build_report(self, raw_df, dedup_df, with_paths_df, clean_df):
        to_dict = lambda s: {str(k): int(v) for k, v in s.to_dict().items()}

        report = {
            "rows_raw": len(raw_df),
            "removed_duplicate_ids": len(raw_df) - len(dedup_df),
            "rows_with_files": len(with_paths_df),
            "label_distribution_raw": to_dict(raw_df["Label"].value_counts()),
        }

        if with_paths_df.empty:
            report["rows_missing_files"] = len(dedup_df)
            report["removed_pixel_duplicates"] = 0
            report["potential_mislabeled_groups"] = []
            report["label_distribution_after_dedup"] = {}
            return report


        report["rows_missing_files"] = len(dedup_df) - len(with_paths_df)
        report["removed_pixel_duplicates"] = len(with_paths_df) - len(clean_df)
        report["potential_mislabeled_groups"] = self._find_mislabeled_groups(with_paths_df)
        report["label_distribution_after_dedup"] = to_dict(clean_df["Label"].value_counts())

        return report


    def analyze(self):
        raw_df, dedup_df = self._load_and_dedup_ids()
        with_paths_df, clean_df = self._clean(dedup_df)

        return self._build_report(raw_df, dedup_df, with_paths_df, clean_df)


    def split(self, return_report=False):
        raw_df, dedup_df = self._load_and_dedup_ids()
        with_paths_df, clean_df = self._clean(dedup_df)

        label_counts = clean_df["Label"].value_counts()
        rare = label_counts[label_counts < 2].index
        if len(rare) > 0:
            print(f"[WARNING] Dropping {clean_df['Label'].isin(rare).sum()} samples (too rare for stratify): {list(rare)}")
            clean_df = clean_df[~clean_df["Label"].isin(rare)].reset_index(drop=True)

        train_df, val_df = train_test_split(
            clean_df,
            test_size=self.val_fraction,
            random_state=self.random_state,
            stratify=clean_df["Label"],
        )
        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)

        if return_report:
            report = self._build_report(raw_df, dedup_df, with_paths_df, clean_df)
            return train_df, val_df, report

        return train_df, val_df
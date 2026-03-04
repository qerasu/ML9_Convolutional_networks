import hashlib
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


class Preprocessor:
    SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}

    def __init__(
        self,
        csv_path="../datasets/Train.csv",
        images_dir="../datasets/Images",
        val_fraction=0.33,
        random_state=42,
    ):
        self.csv_path = Path(csv_path)
        self.images_dir = Path(images_dir)
        self.val_fraction = val_fraction
        self.random_state = random_state


    @staticmethod
    def _to_dict(series):
        return {str(k): int(v) for k, v in series.to_dict().items()}


    # reducing memory pressure
    @staticmethod
    def _file_hash(path):
        digest = hashlib.md5()
        with path.open("rb") as file:
            for chunk in iter(lambda: file.read(8192), b""):
                digest.update(chunk)

        return digest.hexdigest()


    def _load(self):
        df = pd.read_csv(self.csv_path)

        rows_before = df.shape[0]
        df = df.drop_duplicates(subset=["img_IDS"], keep="first").reset_index(drop=True)
        print(f"Removed {rows_before - df.shape[0]} duplicate rows by img_IDS.")

        return df


    def _stem_to_path(self):
        mapping = {}
        for ext in self.SUPPORTED_EXTENSIONS:
            for path in self.images_dir.glob(f"*{ext}"):
                mapping[path.stem] = path
            for path in self.images_dir.glob(f"*{ext.upper()}"):
                mapping[path.stem] = path

        return mapping


    def _attach_paths(self, df):
        df = df.copy()
        df["filepath"] = df["img_IDS"].map(self._stem_to_path())
        return df


    def _find_mislabeled_groups(self, df):
        conflict_hashes = df.groupby("img_hash")["Label"].nunique()
        conflict_hashes = conflict_hashes[conflict_hashes > 1].index

        groups = []
        for hash_value in conflict_hashes:
            group_df = df[df["img_hash"] == hash_value][["img_IDS", "Label"]]
            groups.append(
                {
                    "hash": hash_value,
                    "samples": [
                        {"img_IDS": str(row["img_IDS"]), "Label": int(row["Label"])}
                        for _, row in group_df.iterrows()
                    ],
                }
            )

        return groups


    def _drop_pixel_duplicates(self, df):
        df = self._attach_paths(df)
        missing_files = int(df["filepath"].isna().sum())
        if missing_files:
            print(f"[WARN] {missing_files} rows do not have matching image files and will be dropped.")

        df = df.dropna(subset=["filepath"]).copy()
        if df.empty:
            return df

        df["img_hash"] = df["filepath"].apply(self._file_hash)

        mislabeled_groups = self._find_mislabeled_groups(df)
        if mislabeled_groups:
            print(f"[MISLABEL] Found {len(mislabeled_groups)} potential mislabeled groups.")
        else:
            print("[MISLABEL] No conflicting labels found.")

        rows_before = df.shape[0]
        df = df.drop_duplicates(subset=["img_hash"], keep="first").reset_index(drop=True)
        print(f"[DEDUP] Removed {rows_before - df.shape[0]} pixel-identical duplicates.")

        return df.drop(columns=["filepath", "img_hash"], errors="ignore")


    def analyze(self):
        raw_df = pd.read_csv(self.csv_path)
        dedup_id_df = raw_df.drop_duplicates(subset=["img_IDS"], keep="first").reset_index(drop=True)
        with_paths_df = self._attach_paths(dedup_id_df)
        with_paths_df = with_paths_df.dropna(subset=["filepath"]).copy()

        report = {
            "rows_raw": int(raw_df.shape[0]),
            "removed_duplicate_ids": int(raw_df.shape[0] - dedup_id_df.shape[0]),
            "rows_with_files": int(with_paths_df.shape[0]),
            "label_distribution_raw": self._to_dict(raw_df["Label"].value_counts()),
        }

        if with_paths_df.empty:
            report["rows_missing_files"] = int(dedup_id_df.shape[0])
            report["removed_pixel_duplicates"] = 0
            report["potential_mislabeled_groups"] = []
            report["label_distribution_after_dedup"] = {}
            return report

        with_paths_df["img_hash"] = with_paths_df["filepath"].apply(self._file_hash)
        dedup_hash_df = with_paths_df.drop_duplicates(subset=["img_hash"], keep="first").reset_index(drop=True)
        label_counts = self._to_dict(dedup_hash_df["Label"].value_counts())

        report["rows_missing_files"] = int(dedup_id_df.shape[0] - with_paths_df.shape[0])
        report["removed_pixel_duplicates"] = int(with_paths_df.shape[0] - dedup_hash_df.shape[0])
        report["potential_mislabeled_groups"] = self._find_mislabeled_groups(with_paths_df)
        report["label_distribution_after_dedup"] = label_counts

        return report


    def split(self, return_report=False):
        df = self._load()
        df = self._drop_pixel_duplicates(df)

        train_df, val_df = train_test_split(
            df,
            test_size=self.val_fraction,
            random_state=self.random_state,
            stratify=df["Label"],
        )
        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)

        if return_report:
            return train_df, val_df, self.analyze()

        return train_df, val_df
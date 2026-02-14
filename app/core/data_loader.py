"""Optimized data loading with format detection and error recovery."""

from typing import Optional
from io import BytesIO
import pandas as pd
import streamlit as st


class DataLoader:
    """Centralized data loader with caching and error handling."""

    SUPPORTED_FORMATS = {
        "csv": [".csv", ".tsv"],
        "excel": [".xlsx", ".xls", ".xlsm"],
        "json": [".json", ".jsonl"],
        "parquet": [".parquet", ".pq"],
    }

    @staticmethod
    def get_sheet_names(file_bytes: bytes) -> list[str]:
        """Get sheet names from an Excel file."""
        try:
            return pd.ExcelFile(BytesIO(file_bytes), engine="openpyxl").sheet_names
        except Exception:
            return []

    @staticmethod
    @st.cache_data(show_spinner="Loading data...")
    def load(file_bytes: bytes, filename: str, sheet_name: Optional[str] = None) -> Optional[pd.DataFrame]:
        """Load any supported file format with automatic detection."""
        bio = BytesIO(file_bytes)
        name = filename.lower()

        loaders = [
            (DataLoader._is_csv, DataLoader._load_csv),
            (DataLoader._is_excel, DataLoader._load_excel),
            (DataLoader._is_json, DataLoader._load_json),
            (DataLoader._is_parquet, DataLoader._load_parquet),
        ]

        for check_fn, load_fn in loaders:
            if check_fn(name):
                try:
                    bio.seek(0)
                    if name.endswith(tuple(DataLoader.SUPPORTED_FORMATS["excel"])):
                        return load_fn(bio, name, sheet_name)
                    return load_fn(bio, name)
                except Exception as e:
                    st.warning(f"Failed with {load_fn.__name__}: {e}")
                    continue

        bio.seek(0)
        return DataLoader._auto_detect_and_load(bio, name)

    # ── format checkers ──────────────────────────────────────────────
    @staticmethod
    def _is_csv(n: str) -> bool:
        return any(n.endswith(e) for e in DataLoader.SUPPORTED_FORMATS["csv"])

    @staticmethod
    def _is_excel(n: str) -> bool:
        return any(n.endswith(e) for e in DataLoader.SUPPORTED_FORMATS["excel"])

    @staticmethod
    def _is_json(n: str) -> bool:
        return any(n.endswith(e) for e in DataLoader.SUPPORTED_FORMATS["json"])

    @staticmethod
    def _is_parquet(n: str) -> bool:
        return any(n.endswith(e) for e in DataLoader.SUPPORTED_FORMATS["parquet"])

    # ── loaders ──────────────────────────────────────────────────────
    @staticmethod
    def _load_csv(bio: BytesIO, name: str) -> pd.DataFrame:
        delimiter = "\t" if name.endswith(".tsv") else ","
        try:
            return pd.read_csv(bio, delimiter=delimiter)
        except Exception:
            bio.seek(0)
            return pd.read_csv(bio, sep=None, engine="python")

    @staticmethod
    def _load_excel(bio: BytesIO, _name: str, sheet_name: Optional[str] = None) -> pd.DataFrame:
        return pd.read_excel(bio, engine="openpyxl", sheet_name=sheet_name)

    @staticmethod
    def _load_json(bio: BytesIO, name: str) -> pd.DataFrame:
        if name.endswith(".jsonl"):
            return pd.read_json(bio, lines=True)
        try:
            return pd.read_json(bio)
        except ValueError:
            bio.seek(0)
            return pd.read_json(bio, lines=True)

    @staticmethod
    def _load_parquet(bio: BytesIO, _name: str) -> pd.DataFrame:
        return pd.read_parquet(bio)

    @staticmethod
    def _auto_detect_and_load(bio: BytesIO, _name: str) -> pd.DataFrame:
        """Try to auto-detect format from content."""
        bio.seek(0)
        header = bio.read(100)
        bio.seek(0)

        if b"," in header or b"\t" in header or b"\n" in header:
            try:
                return pd.read_csv(bio, sep=None, engine="python")
            except Exception:
                pass

        if header.startswith(b"PK"):
            try:
                bio.seek(0)
                return pd.read_excel(bio)
            except Exception:
                pass

        if header.startswith(b"{") or header.startswith(b"["):
            try:
                bio.seek(0)
                return pd.read_json(bio)
            except Exception:
                bio.seek(0)
                return pd.read_json(bio, lines=True)

        bio.seek(0)
        return pd.read_csv(bio)


def validate_dataframe(df: pd.DataFrame) -> tuple[bool, str]:
    """Validate loaded dataframe and return (is_valid, message)."""
    if df is None:
        return False, "Failed to load data"
    if df.empty:
        return False, "Dataset is empty"
    if len(df.columns) == 0:
        return False, "No columns found in dataset"
    if len(df) == 0:
        return False, "No rows found in dataset"
    if len(df) > 1_000_000:
        st.warning(f"Large dataset: {len(df):,} rows. Performance may be impacted.")
    if len(df.columns) > 1000:
        st.warning(f"Many columns: {len(df.columns):,}. Consider feature selection.")
    return True, f"Loaded {len(df):,} rows × {len(df.columns)} columns"

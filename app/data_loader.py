"""Optimized data loading module with format detection and error recovery."""

from typing import Optional
from io import BytesIO
import pandas as pd
import streamlit as st


class DataLoader:
    """Centralized data loader with caching and error handling."""
    
    SUPPORTED_FORMATS = {
        'csv': ['.csv', '.tsv'],
        'excel': ['.xlsx', '.xls', '.xlsm'],
        'json': ['.json', '.jsonl'],
        'parquet': ['.parquet', '.pq']
    }
    
    @staticmethod
    @st.cache_data(show_spinner="Loading data...")
    def load(file_bytes: bytes, filename: str) -> Optional[pd.DataFrame]:
        """Load any supported file format with automatic detection."""
        
        bio = BytesIO(file_bytes)
        name = filename.lower()
        
        # Try format-specific loaders
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
                    return load_fn(bio, name)
                except Exception as e:
                    st.warning(f"Failed with {load_fn.__name__}: {e}")
                    continue
        
        # Fallback: try to detect format from content
        bio.seek(0)
        return DataLoader._auto_detect_and_load(bio, name)
    
    @staticmethod
    def _is_csv(name: str) -> bool:
        return any(name.endswith(ext) for ext in DataLoader.SUPPORTED_FORMATS['csv'])
    
    @staticmethod
    def _is_excel(name: str) -> bool:
        return any(name.endswith(ext) for ext in DataLoader.SUPPORTED_FORMATS['excel'])
    
    @staticmethod
    def _is_json(name: str) -> bool:
        return any(name.endswith(ext) for ext in DataLoader.SUPPORTED_FORMATS['json'])
    
    @staticmethod
    def _is_parquet(name: str) -> bool:
        return any(name.endswith(ext) for ext in DataLoader.SUPPORTED_FORMATS['parquet'])
    
    @staticmethod
    def _load_csv(bio: BytesIO, name: str) -> pd.DataFrame:
        """Load CSV/TSV with intelligent delimiter detection."""
        
        delimiter = '\t' if name.endswith('.tsv') else ','
        
        # Try with inferred delimiter
        try:
            return pd.read_csv(bio, delimiter=delimiter)
        except Exception:
            # Try auto-detecting delimiter
            bio.seek(0)
            return pd.read_csv(bio, sep=None, engine='python')
    
    @staticmethod
    def _load_excel(bio: BytesIO, name: str) -> pd.DataFrame:
        """Load Excel files."""
        
        return pd.read_excel(bio, engine='openpyxl')
    
    @staticmethod
    def _load_json(bio: BytesIO, name: str) -> pd.DataFrame:
        """Load JSON/JSONL files."""
        
        if name.endswith('.jsonl'):
            return pd.read_json(bio, lines=True)
        else:
            # Try normal JSON first, then lines format
            try:
                return pd.read_json(bio)
            except ValueError:
                bio.seek(0)
                return pd.read_json(bio, lines=True)
    
    @staticmethod
    def _load_parquet(bio: BytesIO, name: str) -> pd.DataFrame:
        """Load Parquet files."""
        
        return pd.read_parquet(bio)
    
    @staticmethod
    def _auto_detect_and_load(bio: BytesIO, name: str) -> pd.DataFrame:
        """Try to auto-detect format from content."""
        
        # Read first few bytes to detect format
        bio.seek(0)
        header = bio.read(100)
        bio.seek(0)
        
        # Check for common patterns
        if b',' in header or b'\t' in header or b'\n' in header:
            # Likely CSV/TSV
            try:
                return pd.read_csv(bio, sep=None, engine='python')
            except Exception:
                pass
        
        if header.startswith(b'PK'):
            # Likely Excel (ZIP format)
            try:
                bio.seek(0)
                return pd.read_excel(bio)
            except Exception:
                pass
        
        if header.startswith(b'{') or header.startswith(b'['):
            # Likely JSON
            try:
                bio.seek(0)
                return pd.read_json(bio)
            except Exception:
                bio.seek(0)
                return pd.read_json(bio, lines=True)
        
        # Last resort: try CSV
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
    
    # Check for reasonable size
    if len(df) > 1_000_000:
        st.warning(f"Large dataset: {len(df):,} rows. Performance may be impacted.")
    
    if len(df.columns) > 1000:
        st.warning(f"Many columns: {len(df.columns):,}. Consider feature selection.")
    
    return True, f"Loaded {len(df):,} rows × {len(df.columns)} columns"


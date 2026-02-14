"""
Lightweight RAG over user-uploaded DataFrames.

Uses TF-IDF cosine similarity (scikit-learn) to retrieve the most
relevant data context for a user question — zero new dependencies.
"""

from __future__ import annotations
from typing import List, Optional
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class DataRAG:
    """Build knowledge chunks from a DataFrame and retrieve by question."""

    def __init__(self, df: pd.DataFrame, max_chunks: int = 200):
        self.df = df
        self._chunks: List[str] = []
        self._vectorizer: Optional[TfidfVectorizer] = None
        self._tfidf_matrix = None
        self._build_chunks(max_chunks)
        self._fit()

    # ── chunk builders ───────────────────────────────────────────────
    def _build_chunks(self, max_chunks: int):
        """Build structured text chunks from the DataFrame."""
        chunks: List[str] = []

        # 1. Dataset overview
        chunks.append(
            f"DATASET OVERVIEW: {self.df.shape[0]} rows, {self.df.shape[1]} columns. "
            f"Columns: {', '.join(self.df.columns.tolist())}. "
            f"Memory usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.1f} MB. "
            f"Missing cells: {self.df.isnull().sum().sum()} / {self.df.shape[0] * self.df.shape[1]}."
        )

        num_cols = self.df.select_dtypes(include="number").columns.tolist()
        cat_cols = self.df.select_dtypes(include=["object", "category"]).columns.tolist()
        dt_cols = self.df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()

        # 2. Column-level stats
        for col in num_cols:
            s = self.df[col]
            chunks.append(
                f"NUMERIC COLUMN '{col}': "
                f"type={s.dtype}, count={s.count()}, missing={s.isnull().sum()}, "
                f"mean={s.mean():.4g}, median={s.median():.4g}, "
                f"std={s.std():.4g}, min={s.min():.4g}, max={s.max():.4g}, "
                f"25%={s.quantile(0.25):.4g}, 75%={s.quantile(0.75):.4g}, "
                f"skew={s.skew():.2f}, unique={s.nunique()}."
            )

        for col in cat_cols:
            s = self.df[col]
            top5 = s.value_counts().head(5)
            dist = ", ".join(f"{k}={v}" for k, v in top5.items())
            chunks.append(
                f"CATEGORICAL COLUMN '{col}': "
                f"unique={s.nunique()}, missing={s.isnull().sum()}, "
                f"top values: [{dist}]."
            )

        for col in dt_cols:
            s = self.df[col].dropna()
            if len(s) > 0:
                chunks.append(
                    f"DATETIME COLUMN '{col}': "
                    f"range {s.min()} to {s.max()}, "
                    f"missing={self.df[col].isnull().sum()}."
                )

        # 3. Correlations (top pairs)
        if len(num_cols) >= 2:
            corr = self.df[num_cols].corr()
            pairs = []
            for i in range(len(num_cols)):
                for j in range(i + 1, len(num_cols)):
                    val = corr.iloc[i, j]
                    if abs(val) > 0.3:
                        pairs.append((num_cols[i], num_cols[j], val))
            pairs.sort(key=lambda x: abs(x[2]), reverse=True)
            for a, b, v in pairs[:15]:
                strength = "strong" if abs(v) > 0.7 else "moderate"
                direction = "positive" if v > 0 else "negative"
                chunks.append(f"CORRELATION: '{a}' and '{b}' have {strength} {direction} correlation ({v:.3f}).")

        # 4. Sample rows
        head = self.df.head(3).to_string(index=False)
        chunks.append(f"SAMPLE ROWS (first 3):\n{head}")
        if len(self.df) > 5:
            tail = self.df.tail(2).to_string(index=False)
            chunks.append(f"SAMPLE ROWS (last 2):\n{tail}")

        # 5. Duplicates & quality
        dups = self.df.duplicated().sum()
        if dups > 0:
            chunks.append(f"DATA QUALITY: {dups} duplicate rows found ({dups/len(self.df)*100:.1f}%).")

        self._chunks = chunks[:max_chunks]

    def _fit(self):
        """Fit TF-IDF vectorizer on chunks."""
        if not self._chunks:
            return
        self._vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
        self._tfidf_matrix = self._vectorizer.fit_transform(self._chunks)

    # ── retrieval ────────────────────────────────────────────────────
    def retrieve(self, question: str, top_k: int = 8) -> List[str]:
        """Retrieve the top-k most relevant chunks for a question."""
        if self._vectorizer is None or self._tfidf_matrix is None:
            return self._chunks[:top_k]

        q_vec = self._vectorizer.transform([question])
        scores = cosine_similarity(q_vec, self._tfidf_matrix).flatten()
        # always include the dataset overview (chunk 0)
        top_indices = np.argsort(scores)[::-1]
        selected = [0] if 0 not in top_indices[:top_k] else []
        for idx in top_indices:
            if len(selected) >= top_k:
                break
            if idx not in selected:
                selected.append(idx)
        return [self._chunks[i] for i in sorted(selected)]

    def build_context(self, question: str, top_k: int = 8) -> str:
        """Build a context string for the LLM from retrieved chunks."""
        chunks = self.retrieve(question, top_k)
        return "\n\n".join(chunks)

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)

# elite_financial_platform_v6.py
# Enterprise Financial Analytics Platform v6.0 — Complete Rewrite
# Designed for Capitaline financial data with Penman-Nissim framework
#
# Architecture: Clean single-file with logical sections
# Zero redundancy, improved PN analysis, robust parsing
#
# Section Map:
#   1. Imports & Constants
#   2. Configuration
#   3. Logging & Performance
#   4. Caching
#   5. Year Detection & Parsing Utilities
#   6. Capitaline Parser
#   7. Data Cleaning & Validation
#   8. Metric Pattern Matching & Mapping
#   9. Kaggle API Client (Simplified)
#  10. AI/Fuzzy Mapper
#  11. Financial Analysis Engine
#  12. Penman-Nissim Analyzer (Complete Rewrite)
#  13. ML Forecasting
#  14. Number Formatting & Export
#  15. UI Components & Rendering
#  16. Main Application Class
#  17. Entry Point

# ═══════════════════════════════════════════════════════════
# SECTION 1: IMPORTS & CONSTANTS
# ═══════════════════════════════════════════════════════════

import functools
import gc
import hashlib
import importlib
import io
import json
import logging
import os
import re
import sys
import tempfile
import shutil
import time
import traceback
import warnings
import zipfile
from abc import ABC, abstractmethod
from collections import defaultdict, OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from functools import lru_cache
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics.pairwise import cosine_similarity

import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

import bleach
from fuzzywuzzy import fuzz

warnings.filterwarnings('ignore')

# Optional imports
try:
    import py7zr; SEVEN_ZIP_OK = True
except ImportError:
    SEVEN_ZIP_OK = False

try:
    from sentence_transformers import SentenceTransformer; ST_OK = True
except ImportError:
    ST_OK = False

try:
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    REQUESTS_OK = True
except ImportError:
    REQUESTS_OK = False

EPS = 1e-10
YEAR_RE = re.compile(r'(20\d{2}|19\d{2})')
YYYYMM_RE = re.compile(r'(\d{6})')
MAX_FILE_MB = 50


# ═══════════════════════════════════════════════════════════
# SECTION 2: CONFIGURATION
# ═══════════════════════════════════════════════════════════

class Cfg:
    """Minimal, flat configuration. No over-abstraction."""

    # App
    VERSION = '6.0.0'
    DEBUG = False
    ALLOWED_TYPES = ['csv', 'html', 'htm', 'xls', 'xlsx', 'zip', '7z']

    # Analysis
    CONFIDENCE_THRESHOLD = 0.6
    OUTLIER_STD = 3
    MIN_DATA_POINTS = 3

    # AI / Kaggle
    AI_ENABLED = True
    AI_MODEL = 'all-MiniLM-L6-v2'
    KAGGLE_URL = ''
    KAGGLE_TIMEOUT = 30
    KAGGLE_RETRIES = 3
    KAGGLE_BATCH = 50

    # Display
    NUMBER_FORMAT = 'Indian'  # or 'International'

    @classmethod
    def from_session(cls):
        """Apply session-state overrides."""
        cls.KAGGLE_URL = st.session_state.get('kaggle_api_url', '')
        cls.NUMBER_FORMAT = st.session_state.get('number_format', 'Indian')
        cls.DEBUG = st.session_state.get('debug_mode', False)


# ═══════════════════════════════════════════════════════════
# SECTION 3: LOGGING & PERFORMANCE
# ═══════════════════════════════════════════════════════════

class Log:
    """Centralised logger factory. One logger per name, no duplicates."""
    _cache: Dict[str, logging.Logger] = {}
    _dir = Path("logs")

    @classmethod
    def get(cls, name: str) -> logging.Logger:
        if name in cls._cache:
            return cls._cache[name]
        cls._dir.mkdir(exist_ok=True)
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG if Cfg.DEBUG else logging.INFO)
        logger.handlers.clear()
        fmt = logging.Formatter('%(asctime)s|%(name)s|%(levelname)s|%(message)s')
        sh = logging.StreamHandler(); sh.setFormatter(fmt)
        fh = RotatingFileHandler(cls._dir / f"{name}.log", maxBytes=5_000_000, backupCount=3)
        fh.setFormatter(fmt)
        logger.addHandler(sh); logger.addHandler(fh)
        cls._cache[name] = logger
        return logger


class Perf:
    """Lightweight performance tracker."""
    _data: Dict[str, List[float]] = defaultdict(list)

    @classmethod
    @contextmanager
    def measure(cls, name: str):
        t0 = time.time()
        yield
        cls._data[name].append(time.time() - t0)

    @classmethod
    def summary(cls) -> Dict[str, Dict]:
        return {k: {'avg': np.mean(v), 'count': len(v), 'total': sum(v)}
                for k, v in cls._data.items() if v}


# ═══════════════════════════════════════════════════════════
# SECTION 4: CACHING
# ═══════════════════════════════════════════════════════════

class Cache:
    """Simple TTL cache with LRU eviction. No compression over-engineering."""

    def __init__(self, max_entries: int = 500, ttl: int = 3600):
        self._store: OrderedDict = OrderedDict()
        self._max = max_entries
        self._ttl = ttl
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Optional[Any]:
        if key in self._store:
            entry = self._store[key]
            if time.time() - entry['ts'] < self._ttl:
                self._store.move_to_end(key)
                self._hits += 1
                return entry['val']
            del self._store[key]
        self._misses += 1
        return None

    def put(self, key: str, value: Any):
        self._store[key] = {'val': value, 'ts': time.time()}
        self._store.move_to_end(key)
        while len(self._store) > self._max:
            self._store.popitem(last=False)

    def clear(self):
        self._store.clear()

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return (self._hits / total * 100) if total else 0


# ═══════════════════════════════════════════════════════════
# SECTION 5: YEAR DETECTION & PARSING UTILITIES
# ═══════════════════════════════════════════════════════════

class YearDetector:
    """Detect and normalise year columns from various formats."""

    PATTERNS = [
        (re.compile(r'^(\d{6})$'), lambda m: m.group(1)),              # 202003
        (re.compile(r'(\d{4})(\d{2})'), lambda m: m.group(0)),         # embedded YYYYMM
        (re.compile(r'FY\s*(\d{4})'), lambda m: m.group(1) + '03'),   # FY2023
        (re.compile(r'Mar(?:ch)?[- ]?(\d{4})'), lambda m: m.group(1) + '03'),
        (re.compile(r'Mar(?:ch)?[- ]?(\d{2})$'), lambda m: '20' + m.group(1) + '03'),
        (re.compile(r'(\d{4})-(\d{2})'), lambda m: m.group(1) + '03'),  # 2023-24
        (re.compile(r'(20\d{2}|19\d{2})'), lambda m: m.group(1) + '03'),  # plain 2023
    ]

    @classmethod
    def extract(cls, col_name: str) -> Optional[str]:
        """Extract normalised YYYYMM from column name."""
        s = str(col_name).strip()
        for pat, extractor in cls.PATTERNS:
            m = pat.search(s)
            if m:
                result = extractor(m)
                year = int(result[:4])
                if 1990 <= year <= 2099:
                    return result
        return None

    @classmethod
    def detect_columns(cls, df: pd.DataFrame) -> Dict[str, List]:
        """Map normalised years → list of original column names."""
        mapping = defaultdict(list)
        for col in df.columns:
            year = cls.extract(col)
            if year:
                mapping[year].append(col)
        return dict(mapping)


class StatementClassifier:
    """Classify financial line items by statement type."""

    PL_KW = {'revenue', 'sales', 'income', 'profit', 'loss', 'expense', 'cost',
             'ebit', 'ebitda', 'tax', 'interest', 'depreciation', 'amortisation',
             'amortization', 'dividend', 'earning', 'margin', 'turnover'}
    BS_KW = {'asset', 'liability', 'liabilities', 'equity', 'capital', 'reserve',
             'surplus', 'receivable', 'payable', 'inventory', 'inventories',
             'borrowing', 'debt', 'investment', 'property', 'plant', 'goodwill',
             'cash', 'bank', 'provision', 'debenture'}
    CF_KW = {'cash flow', 'operating activities', 'investing activities',
             'financing activities', 'capex', 'capital expenditure',
             'purchase of fixed', 'net cash'}

    @classmethod
    def classify(cls, text: str) -> str:
        """Classify a single metric name → BalanceSheet|ProfitLoss|CashFlow|Financial."""
        low = text.lower()
        # Prefix check first
        if low.startswith('profitloss::') or low.startswith('p&l::'):
            return 'ProfitLoss'
        if low.startswith('balancesheet::') or low.startswith('bs::'):
            return 'BalanceSheet'
        if low.startswith('cashflow::') or low.startswith('cf::'):
            return 'CashFlow'
        # Keyword scoring
        scores = {'ProfitLoss': 0, 'BalanceSheet': 0, 'CashFlow': 0}
        for kw in cls.CF_KW:
            if kw in low:
                scores['CashFlow'] += 3
        for kw in cls.PL_KW:
            if kw in low:
                scores['ProfitLoss'] += 1
        for kw in cls.BS_KW:
            if kw in low:
                scores['BalanceSheet'] += 1
        best = max(scores, key=scores.get)
        return best if scores[best] > 0 else 'Financial'

    @classmethod
    def classify_table(cls, df: pd.DataFrame) -> str:
        """Classify an entire table by content."""
        blob = ' '.join(str(v) for v in df.values.flatten()[:300] if pd.notna(v)).lower()
        scores = {
            'CashFlow': sum(1 for kw in cls.CF_KW if kw in blob) * 3,
            'ProfitLoss': sum(1 for kw in cls.PL_KW if kw in blob),
            'BalanceSheet': sum(1 for kw in cls.BS_KW if kw in blob),
        }
        return max(scores, key=scores.get) if max(scores.values()) > 0 else 'Financial'

    @classmethod
    def classify_sheet(cls, sheet_name: str, df: pd.DataFrame) -> str:
        """Classify by sheet name first, then content."""
        s = sheet_name.lower()
        if any(k in s for k in ['cash', 'flow', 'cf']):
            return 'CashFlow'
        if any(k in s for k in ['profit', 'loss', 'p&l', 'pl', 'income']):
            return 'ProfitLoss'
        if any(k in s for k in ['balance', 'bs', 'position']):
            return 'BalanceSheet'
        return cls.classify_table(df)


# ═══════════════════════════════════════════════════════════
# SECTION 6: CAPITALINE PARSER
# ═══════════════════════════════════════════════════════════

class CapitalineParser:
    """
    Parse Capitaline financial exports (.xls as HTML, .html, .xlsx, .csv).
    Handles multi-statement files, year detection, and statement prefixing.
    """

    def __init__(self):
        self.log = Log.get('Parser')

    def parse(self, file) -> Optional[pd.DataFrame]:
        """Main entry: detect format and parse."""
        name = getattr(file, 'name', 'unknown')
        ext = Path(name).suffix.lower()
        file.seek(0)
        self.log.info(f"Parsing {name} (ext={ext})")

        try:
            if ext in ('.html', '.htm', '.xls'):
                return self._parse_html_tables(file, name)
            elif ext == '.xlsx':
                return self._parse_xlsx(file, name)
            elif ext == '.csv':
                return self._parse_csv(file, name)
            else:
                self.log.error(f"Unsupported extension: {ext}")
                return None
        except Exception as e:
            self.log.error(f"Parse error for {name}: {e}", exc_info=True)
            return None

    # ── HTML / XLS-as-HTML ──

    def _parse_html_tables(self, file, name: str) -> Optional[pd.DataFrame]:
        """Parse HTML containing multiple financial tables."""
        file.seek(0)
        try:
            tables = pd.read_html(file, header=None)
        except Exception as e:
            self.log.error(f"pd.read_html failed: {e}")
            return None

        if not tables:
            self.log.error("No HTML tables found")
            return None

        self.log.info(f"Found {len(tables)} tables in {name}")
        processed = []

        for i, tbl in enumerate(tables):
            if tbl.shape[0] < 5 or tbl.shape[1] < 2:
                continue
            df = self._process_html_table(tbl, i)
            if df is not None and not df.empty:
                processed.append(df)

        if not processed:
            # Fallback: use the largest table
            largest = max(tables, key=len)
            df = self._process_html_table(largest, 0)
            if df is not None:
                processed.append(df)

        if processed:
            result = pd.concat(processed, axis=0)
            result = result[~result.index.duplicated(keep='first')]
            self.log.info(f"HTML parse complete: {result.shape}")
            return result
        return None

    def _process_html_table(self, tbl: pd.DataFrame, idx: int) -> Optional[pd.DataFrame]:
        """Process a single HTML table: find header row, extract data, add prefixes."""
        header_row = self._find_header_row(tbl)
        if header_row is None:
            return None

        stmt_type = StatementClassifier.classify_table(tbl)

        # Extract headers and data
        headers = tbl.iloc[header_row].tolist()
        data = tbl.iloc[header_row + 1:].copy()
        data.columns = headers

        # Find the metric-name column (first text column)
        metric_col = self._find_metric_column(data)
        if metric_col is not None:
            data = data.set_index(data.columns[metric_col])

        # Normalise year columns
        data = self._normalise_columns(data)

        # Convert to numeric
        for col in data.columns:
            data[col] = self._to_numeric(data[col])

        data = data.dropna(how='all').dropna(axis=1, how='all')

        # Add statement prefix
        data.index = [f"{stmt_type}::{str(idx_val).strip()}" for idx_val in data.index]

        self.log.info(f"Table {idx}: {stmt_type}, {len(data)} rows, cols={list(data.columns)}")
        return data

    # ── XLSX ──

    def _parse_xlsx(self, file, name: str) -> Optional[pd.DataFrame]:
        """Parse Excel with multiple sheets."""
        file.seek(0)
        xl = pd.ExcelFile(file)
        all_dfs = []

        for sheet in xl.sheet_names:
            try:
                raw = pd.read_excel(file, sheet_name=sheet, header=None)
                if raw.empty or raw.shape[0] < 3:
                    continue

                stmt_type = StatementClassifier.classify_sheet(sheet, raw)
                header_row = self._find_header_row(raw)

                if header_row is not None:
                    headers = raw.iloc[header_row].tolist()
                    data = raw.iloc[header_row + 1:].copy()
                    data.columns = headers
                    metric_col = self._find_metric_column(data)
                    if metric_col is not None:
                        data = data.set_index(data.columns[metric_col])
                else:
                    data = pd.read_excel(file, sheet_name=sheet)

                data = self._normalise_columns(data)
                for col in data.columns:
                    data[col] = self._to_numeric(data[col])
                data = data.dropna(how='all').dropna(axis=1, how='all')
                data.index = [f"{stmt_type}::{str(v).strip()}" for v in data.index]

                if not data.empty:
                    all_dfs.append(data)
                    self.log.info(f"Sheet '{sheet}': {stmt_type}, {len(data)} rows")

            except Exception as e:
                self.log.warning(f"Sheet '{sheet}' error: {e}")

        if all_dfs:
            result = pd.concat(all_dfs, axis=0)
            result = result[~result.index.duplicated(keep='first')]
            return result
        return None

    # ── CSV ──

    def _parse_csv(self, file, name: str) -> Optional[pd.DataFrame]:
        """Parse CSV with auto-detection of separator and header."""
        file.seek(0)
        for sep in [',', ';', '\t', None]:
            try:
                file.seek(0)
                kw = {'sep': sep, 'header': 0, 'index_col': 0}
                if sep is None:
                    kw['engine'] = 'python'
                df = pd.read_csv(file, **kw)
                if df is not None and not df.empty and len(df.columns) > 0:
                    df = self._normalise_columns(df)
                    for col in df.columns:
                        df[col] = self._to_numeric(df[col])
                    df = df.dropna(how='all').dropna(axis=1, how='all')
                    # Add statement prefixes by content
                    new_idx = []
                    for idx_val in df.index:
                        stype = StatementClassifier.classify(str(idx_val))
                        prefix = f"{stype}::" if not str(idx_val).startswith(f"{stype}::") else ""
                        new_idx.append(f"{prefix}{str(idx_val).strip()}")
                    df.index = new_idx
                    self.log.info(f"CSV parsed: {df.shape}")
                    return df
            except Exception:
                continue
        return None

    # ── Shared Utilities ──

    def _find_header_row(self, df: pd.DataFrame) -> Optional[int]:
        """Find the row containing year patterns (header row)."""
        for i in range(min(20, len(df))):
            row = df.iloc[i]
            year_count = sum(1 for v in row if pd.notna(v) and
                           (YearDetector.extract(str(v)) is not None))
            if year_count >= 2:
                return i
        return None

    def _find_metric_column(self, df: pd.DataFrame) -> Optional[int]:
        """Find the column containing text metric names (usually first text column)."""
        for i in range(min(5, len(df.columns))):
            sample = df.iloc[:min(10, len(df)), i]
            text_count = sum(1 for v in sample if pd.notna(v) and isinstance(v, str) and len(v.strip()) > 2)
            if text_count >= 3:
                return i
        return 0

    def _normalise_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalise column names to YYYYMM format where possible."""
        new_cols = []
        for col in df.columns:
            year = YearDetector.extract(str(col))
            new_cols.append(year if year else str(col).strip())
        df.columns = new_cols
        # Drop non-year columns (keep only normalised year columns)
        year_cols = [c for c in df.columns if re.match(r'^\d{6}$', str(c))]
        if year_cols:
            return df[year_cols]
        return df

    @staticmethod
    def _to_numeric(series: pd.Series) -> pd.Series:
        """Convert a series to numeric, handling Indian accounting formats."""
        if series.dtype == 'object' or series.dtype == 'str':
            cleaned = (series.astype(str)
                      .str.replace(',', '', regex=False)
                      .str.replace('₹', '', regex=False)
                      .str.replace('$', '', regex=False)
                      .str.replace('(', '-', regex=False)
                      .str.replace(')', '', regex=False)
                      .replace({'-': np.nan, '--': np.nan, 'NA': np.nan,
                               'N/A': np.nan, 'nil': '0', 'Nil': '0', '': np.nan}))
            return pd.to_numeric(cleaned, errors='coerce')
        return pd.to_numeric(series, errors='coerce')


class CompressedFileHandler:
    """Extract ZIP/7Z and return list of (name, bytes) tuples."""

    def __init__(self):
        self.log = Log.get('Compression')
        self._temp_dirs: List[Path] = []

    def extract(self, file: UploadedFile) -> List[Tuple[str, bytes]]:
        result = []
        tmp = Path(tempfile.mkdtemp())
        self._temp_dirs.append(tmp)
        try:
            tmp_file = tmp / file.name
            with open(tmp_file, 'wb') as f:
                f.write(file.getbuffer())

            supported = ('.csv', '.html', '.htm', '.xls', '.xlsx')

            if file.name.lower().endswith('.zip'):
                with zipfile.ZipFile(tmp_file) as zf:
                    for name in zf.namelist():
                        if name.endswith('/') or name.startswith('.'):
                            continue
                        if any(name.lower().endswith(e) for e in supported):
                            result.append((Path(name).name, zf.read(name)))

            elif file.name.lower().endswith('.7z') and SEVEN_ZIP_OK:
                with py7zr.SevenZipFile(tmp_file, 'r') as sz:
                    sz.extractall(tmp)
                for p in tmp.rglob('*'):
                    if p.is_file() and any(p.name.lower().endswith(e) for e in supported):
                        result.append((p.name, p.read_bytes()))

        except Exception as e:
            self.log.error(f"Extraction error: {e}")
        return result

    def cleanup(self):
        for d in self._temp_dirs:
            shutil.rmtree(d, ignore_errors=True)
        self._temp_dirs.clear()


# ═══════════════════════════════════════════════════════════
# SECTION 7: DATA CLEANING & VALIDATION
# ═══════════════════════════════════════════════════════════

@dataclass
class ValidationReport:
    """Validation result container."""
    valid: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    corrections: List[str] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)

    def add_error(self, msg: str):
        self.errors.append(msg)
        self.valid = False


class DataCleaner:
    """Clean, validate, and prepare financial DataFrames."""

    def __init__(self):
        self.log = Log.get('Cleaner')

    def clean(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, ValidationReport]:
        """Full cleaning pipeline."""
        report = ValidationReport()
        if df.empty:
            report.add_error("DataFrame is empty")
            return df, report

        out = df.copy()

        # 1. Deduplicate index
        if out.index.duplicated().any():
            dups = out.index.duplicated().sum()
            report.warnings.append(f"{dups} duplicate indices found, keeping first")
            out = out[~out.index.duplicated(keep='first')]

        # 2. Convert all columns to numeric
        for col in out.columns:
            out[col] = CapitalineParser._to_numeric(out[col])

        # 3. Remove fully empty rows/columns
        before = out.shape
        out = out.dropna(how='all').dropna(axis=1, how='all')
        if out.shape != before:
            report.corrections.append(f"Removed empty: {before} → {out.shape}")

        # 4. Compute stats
        num = out.select_dtypes(include=[np.number])
        if not num.empty:
            total = num.size
            missing = num.isna().sum().sum()
            report.stats['completeness'] = (1 - missing / total) * 100 if total else 0
            report.stats['shape'] = out.shape
        else:
            report.stats['completeness'] = 0

        return out, report

    def validate_file(self, file: UploadedFile) -> ValidationReport:
        """Validate uploaded file (size, extension, security)."""
        report = ValidationReport()
        if file.size > MAX_FILE_MB * 1024 * 1024:
            report.add_error(f"File too large: {file.size / 1e6:.1f}MB > {MAX_FILE_MB}MB")
        ext = Path(file.name).suffix.lower().lstrip('.')
        if ext not in Cfg.ALLOWED_TYPES:
            report.add_error(f"Unsupported type: {ext}")
        suspicious = [r'\.\./', r'[<>"|?*]', r'\.(exe|bat|cmd|sh|ps1)$']
        for pat in suspicious:
            if re.search(pat, file.name, re.I):
                report.add_error("Suspicious filename")
                break
        return report


# ═══════════════════════════════════════════════════════════
# SECTION 8: METRIC PATTERN MATCHING & MAPPING
# ═══════════════════════════════════════════════════════════

class MetricPatterns:
    """
    Centralised metric recognition patterns.
    Each target metric has a list of regex patterns AND a required statement type.
    """

    # (target_name, required_statement, [(pattern, weight), ...])
    REGISTRY: Dict[str, Tuple[str, List[Tuple[re.Pattern, float]]]] = {}

    @classmethod
    def _build(cls):
        if cls.REGISTRY:
            return
        defs = {
            'Total Assets': ('BalanceSheet', [
                'total assets', 'total equity and liabilities', 'assets total']),
            'Current Assets': ('BalanceSheet', [
                'current assets', 'total current assets']),
            'Cash and Cash Equivalents': ('BalanceSheet', [
                'cash and cash equivalents', 'cash & cash equivalents',
                'cash and bank', 'liquid funds']),
            'Inventory': ('BalanceSheet', [
                'inventories', 'inventory', 'stock in trade']),
            'Trade Receivables': ('BalanceSheet', [
                'trade receivables', 'sundry debtors', 'accounts receivable', 'debtors']),
            'Property Plant Equipment': ('BalanceSheet', [
                'property plant and equipment', 'fixed assets', 'tangible assets', 'net block']),
            'Total Liabilities': ('BalanceSheet', [
                'total liabilities', 'total non-current liabilities']),
            'Current Liabilities': ('BalanceSheet', [
                'current liabilities', 'total current liabilities']),
            'Accounts Payable': ('BalanceSheet', [
                'trade payables', 'sundry creditors', 'accounts payable']),
            'Short-term Debt': ('BalanceSheet', [
                'short term borrowings', 'current borrowings', 'other current liabilities']),
            'Long-term Debt': ('BalanceSheet', [
                'long term borrowings', 'non-current borrowings', 'other non-current liabilities']),
            'Total Equity': ('BalanceSheet', [
                'total equity', 'shareholders funds', 'equity', 'net worth']),
            'Share Capital': ('BalanceSheet', [
                'share capital', 'equity share capital', 'paid-up capital']),
            'Retained Earnings': ('BalanceSheet', [
                'reserves and surplus', 'retained earnings', 'other equity']),

            'Revenue': ('ProfitLoss', [
                'revenue from operations', 'revenue from operations(net)',
                'total revenue', 'net sales', 'sales', 'revenue', 'turnover']),
            'Cost of Goods Sold': ('ProfitLoss', [
                'cost of materials consumed', 'cost of goods sold', 'cogs',
                'purchase of stock-in-trade', 'cost of sales']),
            'Operating Expenses': ('ProfitLoss', [
                'employee benefit expenses', 'other expenses', 'operating expenses']),
            'Operating Income': ('ProfitLoss', [
                'profit before exceptional items and tax', 'operating profit',
                'ebit', 'profit before interest and tax', 'operating income']),
            'EBIT': ('ProfitLoss', ['ebit', 'operating profit']),
            'Interest Expense': ('ProfitLoss', [
                'finance cost', 'finance costs', 'interest expense',
                'interest and finance charges', 'borrowing costs']),
            'Other Income': ('ProfitLoss', [
                'other income', 'other operating income', 'miscellaneous income']),
            'Income Before Tax': ('ProfitLoss', [
                'profit before tax', 'pbt', 'income before tax']),
            'Tax Expense': ('ProfitLoss', [
                'tax expense', 'tax expenses', 'current tax', 'total tax expense',
                'income tax', 'provision for tax']),
            'Net Income': ('ProfitLoss', [
                'profit after tax', 'profit/loss for the period', 'net profit',
                'pat', 'net income', 'profit for the period']),
            'Depreciation': ('ProfitLoss', [
                'depreciation and amortisation expenses', 'depreciation and amortization',
                'depreciation', 'depreciation & amortisation']),

            'Operating Cash Flow': ('CashFlow', [
                'net cash from operating activities', 'net cashflow from operating activities',
                'operating cash flow', 'cash from operating activities']),
            'Capital Expenditure': ('CashFlow', [
                'purchase of fixed assets', 'purchased of fixed assets',
                'capital expenditure', 'additions to fixed assets',
                'purchase of property plant and equipment',
                'purchase of investments']),
            'Investing Cash Flow': ('CashFlow', [
                'net cash used in investing', 'cash flow from investing']),
            'Financing Cash Flow': ('CashFlow', [
                'net cash used in financing', 'cash flow from financing']),
        }

        for target, (stmt, patterns) in defs.items():
            compiled = [(re.compile(re.escape(p), re.I), 1.0) for p in patterns]
            cls.REGISTRY[target] = (stmt, compiled)

    @classmethod
    def match(cls, metric_name: str) -> List[Tuple[str, float]]:
        """Return [(target_name, confidence), ...] sorted by confidence desc."""
        cls._build()
        clean = metric_name.split('::')[-1].strip().lower() if '::' in metric_name else metric_name.strip().lower()
        results = []
        for target, (stmt, patterns) in cls.REGISTRY.items():
            best_score = 0
            for pat, weight in patterns:
                if pat.search(clean):
                    # Exact match gets higher score
                    score = 0.95 if pat.pattern.replace('\\', '') == clean else 0.75
                    best_score = max(best_score, score * weight)
            if best_score > 0:
                results.append((target, best_score))
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    @classmethod
    def get_required_statement(cls, target: str) -> Optional[str]:
        cls._build()
        if target in cls.REGISTRY:
            return cls.REGISTRY[target][0]
        return None


class MappingTemplate:
    """Pre-built mapping templates for common data sources."""

    @staticmethod
    def create_auto_mapping(source_metrics: List[str]) -> Tuple[Dict[str, str], List[str]]:
        """Create mappings using pattern matching + fuzzy fallback."""
        MetricPatterns._build()
        mappings = {}
        unmapped = []
        used_targets: Set[str] = set()

        # Pass 1: Pattern matching
        for source in source_metrics:
            matches = MetricPatterns.match(source)
            # Also validate statement type
            source_stmt = source.split('::')[0] if '::' in source else None
            for target, conf in matches:
                if target in used_targets:
                    continue
                required_stmt = MetricPatterns.get_required_statement(target)
                if source_stmt and required_stmt and source_stmt != required_stmt:
                    continue
                if conf >= 0.6:
                    mappings[source] = target
                    used_targets.add(target)
                    break

        # Pass 2: Fuzzy matching for remaining
        remaining = [s for s in source_metrics if s not in mappings]
        all_targets = list(MetricPatterns.REGISTRY.keys())
        available_targets = [t for t in all_targets if t not in used_targets]

        for source in remaining:
            clean = source.split('::')[-1].strip() if '::' in source else source.strip()
            best_target, best_score = None, 0
            for target in available_targets:
                score = fuzz.token_sort_ratio(clean.lower(), target.lower()) / 100
                if score > best_score:
                    best_score = score
                    best_target = target
            if best_target and best_score >= 0.75:
                mappings[source] = best_target
                available_targets.remove(best_target)
            else:
                unmapped.append(source)

        return mappings, unmapped


# ═══════════════════════════════════════════════════════════
# SECTION 9: KAGGLE API CLIENT (SIMPLIFIED)
# ═══════════════════════════════════════════════════════════

class KaggleClient:
    """Simplified Kaggle GPU API client. No over-engineered queuing."""

    def __init__(self, base_url: str, timeout: int = 30, retries: int = 3):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.log = Log.get('Kaggle')
        self._session = None
        self._healthy = False
        self._setup(retries)

    def _setup(self, retries: int):
        if not REQUESTS_OK:
            return
        self._session = requests.Session()
        adapter = HTTPAdapter(
            max_retries=Retry(total=retries, backoff_factor=1,
                             status_forcelist=[429, 500, 502, 503, 504]),
            pool_connections=10, pool_maxsize=20)
        self._session.mount("http://", adapter)
        self._session.mount("https://", adapter)
        self._session.headers.update({
            'Content-Type': 'application/json',
            'ngrok-skip-browser-warning': 'true'})

    def health_check(self) -> bool:
        if not self._session:
            return False
        try:
            # Test embed endpoint directly (most reliable)
            r = self._session.post(
                f"{self.base_url}/embed",
                json={'texts': ['test']},
                timeout=10, verify=False)
            if r.status_code == 200:
                data = r.json()
                self._healthy = 'embeddings' in data
                return self._healthy
        except Exception as e:
            self.log.warning(f"Health check failed: {e}")
        self._healthy = False
        return False

    def embed(self, texts: List[str]) -> Optional[List[np.ndarray]]:
        if not self._session or not self._healthy:
            return None
        try:
            # Batch if needed
            results = []
            for i in range(0, len(texts), Cfg.KAGGLE_BATCH):
                batch = texts[i:i + Cfg.KAGGLE_BATCH]
                r = self._session.post(
                    f"{self.base_url}/embed",
                    json={'texts': batch},
                    timeout=self.timeout, verify=False)
                if r.status_code == 200:
                    data = r.json()
                    if 'embeddings' in data:
                        results.extend([np.array(e) for e in data['embeddings']])
                    else:
                        return None
                else:
                    return None
            return results
        except Exception as e:
            self.log.error(f"Embed error: {e}")
            return None

    def close(self):
        if self._session:
            self._session.close()

    @property
    def is_available(self) -> bool:
        return self._healthy


# ═══════════════════════════════════════════════════════════
# SECTION 10: AI / FUZZY MAPPER
# ═══════════════════════════════════════════════════════════

class MetricMapper:
    """Map source metrics to standard targets using AI embeddings or fuzzy matching."""

    def __init__(self):
        self.log = Log.get('Mapper')
        self._model = None
        self._kaggle: Optional[KaggleClient] = None
        self._embed_cache = Cache(max_entries=2000, ttl=7200)
        self._std_embeddings: Dict[str, np.ndarray] = {}

    def initialize(self, kaggle_url: str = ''):
        """Initialize embedding sources."""
        # Kaggle
        if kaggle_url and REQUESTS_OK:
            self._kaggle = KaggleClient(kaggle_url)
            if self._kaggle.health_check():
                self.log.info("Kaggle GPU connected")
            else:
                self.log.warning("Kaggle unavailable, falling back to local")

        # Local model
        if ST_OK and (not self._kaggle or not self._kaggle.is_available):
            try:
                self._model = SentenceTransformer(Cfg.AI_MODEL)
                self.log.info(f"Local model loaded: {Cfg.AI_MODEL}")
            except Exception as e:
                self.log.warning(f"Local model failed: {e}")

        # Pre-compute standard embeddings
        self._precompute()

    def _precompute(self):
        MetricPatterns._build()
        for target in MetricPatterns.REGISTRY:
            emb = self._get_embedding(target.lower())
            if emb is not None:
                self._std_embeddings[target] = emb

    def _get_embedding(self, text: str) -> Optional[np.ndarray]:
        key = hashlib.md5(text.encode()).hexdigest()
        cached = self._embed_cache.get(key)
        if cached is not None:
            return cached

        emb = None
        if self._kaggle and self._kaggle.is_available:
            result = self._kaggle.embed([text])
            if result:
                emb = result[0]

        if emb is None and self._model:
            try:
                emb = self._model.encode(text, convert_to_numpy=True, show_progress_bar=False)
            except Exception:
                pass

        if emb is not None:
            self._embed_cache.put(key, emb)
        return emb

    def map_metrics(self, sources: List[str], threshold: float = 0.6) -> Dict[str, Any]:
        """Map source metrics to standard targets."""
        # If no AI available, use pure pattern + fuzzy
        if not self._model and not (self._kaggle and self._kaggle.is_available):
            mappings, unmapped = MappingTemplate.create_auto_mapping(sources)
            return {
                'mappings': mappings,
                'confidence': {s: 0.8 for s in mappings},
                'unmapped': unmapped,
                'method': 'pattern+fuzzy'}

        # AI mapping with embeddings
        mappings = {}
        confidence = {}
        unmapped = []
        used_targets: Set[str] = set()

        for source in sources:
            clean = source.split('::')[-1].strip() if '::' in source else source.strip()
            src_emb = self._get_embedding(clean.lower())

            if src_emb is None:
                # Fallback to pattern matching
                matches = MetricPatterns.match(source)
                if matches and matches[0][1] >= threshold:
                    t, c = matches[0]
                    if t not in used_targets:
                        mappings[source] = t
                        confidence[source] = c
                        used_targets.add(t)
                        continue
                unmapped.append(source)
                continue

            # Compute similarities
            best_target, best_score = None, 0
            for target, tgt_emb in self._std_embeddings.items():
                if target in used_targets:
                    continue
                sim = float(cosine_similarity(
                    src_emb.reshape(1, -1), tgt_emb.reshape(1, -1))[0, 0])
                if sim > best_score:
                    best_score = sim
                    best_target = target

            if best_target and best_score >= threshold:
                mappings[source] = best_target
                confidence[source] = best_score
                used_targets.add(best_target)
            else:
                unmapped.append(source)

        method = 'kaggle_ai' if (self._kaggle and self._kaggle.is_available) else 'local_ai'
        return {'mappings': mappings, 'confidence': confidence,
                'unmapped': unmapped, 'method': method}

    def get_status(self) -> Dict:
        return {
            'kaggle_ok': self._kaggle.is_available if self._kaggle else False,
            'local_ok': self._model is not None,
            'cache_entries': len(self._embed_cache._store),
            'cache_hit_rate': self._embed_cache.hit_rate}


# ═══════════════════════════════════════════════════════════
# SECTION 11: FINANCIAL ANALYSIS ENGINE
# ═══════════════════════════════════════════════════════════

class FinancialAnalyzer:
    """Standard financial analysis: ratios, trends, anomalies, insights."""

    def __init__(self):
        self.log = Log.get('Analyzer')
        self._cache = Cache(max_entries=20, ttl=3600)

    def analyze(self, df: pd.DataFrame) -> Dict[str, Any]:
        key = hashlib.md5(str(df.shape).encode() + str(df.index[:5].tolist()).encode()).hexdigest()
        cached = self._cache.get(key)
        if cached:
            return cached

        with Perf.measure('full_analysis'):
            result = {
                'summary': self._summary(df),
                'ratios': self._ratios(df),
                'trends': self._trends(df),
                'anomalies': self._anomalies(df),
                'quality_score': self._quality(df),
                'insights': [],
            }
            result['insights'] = self._insights(result)

        self._cache.put(key, result)
        return result

    def _summary(self, df: pd.DataFrame) -> Dict:
        num = df.select_dtypes(include=[np.number])
        total_cells = num.size if not num.empty else 1
        missing = num.isna().sum().sum() if not num.empty else 0
        return {
            'total_metrics': len(df),
            'years_covered': len(num.columns) if not num.empty else 0,
            'year_range': f"{num.columns[0]}–{num.columns[-1]}" if not num.empty and len(num.columns) > 0 else 'N/A',
            'completeness': (1 - missing / total_cells) * 100}

    def _ratios(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Compute standard financial ratios from mapped data."""
        ratios = {}

        def _find(keyword: str) -> Optional[pd.Series]:
            for idx in df.index:
                if keyword.lower() in str(idx).lower():
                    s = df.loc[idx]
                    return s.iloc[0] if isinstance(s, pd.DataFrame) else s
            return None

        def _safe_div(num: Optional[pd.Series], den: Optional[pd.Series]) -> Optional[pd.Series]:
            if num is None or den is None:
                return None
            return num / den.replace(0, np.nan)

        ca, cl = _find('current assets'), _find('current liabilities')
        ta, tl = _find('total assets'), _find('total liabilities')
        te, rev = _find('total equity'), _find('revenue')
        ni, inv = _find('net income'), _find('inventor')
        ebit, ie = _find('ebit'), _find('interest expense')
        cogs, recv = _find('cost of goods'), _find('receivable')

        # Liquidity
        liq = {}
        cr = _safe_div(ca, cl)
        if cr is not None:
            liq['Current Ratio'] = cr
        ratios['Liquidity'] = pd.DataFrame(liq).T if liq else pd.DataFrame()

        # Profitability
        prof = {}
        npm = _safe_div(ni, rev)
        if npm is not None:
            prof['Net Profit Margin %'] = npm * 100
        roa = _safe_div(ni, ta)
        if roa is not None:
            prof['ROA %'] = roa * 100
        roe = _safe_div(ni, te)
        if roe is not None:
            prof['ROE %'] = roe * 100
        ratios['Profitability'] = pd.DataFrame(prof).T if prof else pd.DataFrame()

        # Leverage
        lev = {}
        de = _safe_div(tl, te)
        if de is not None:
            lev['Debt/Equity'] = de
        icr = _safe_div(ebit, ie)
        if icr is not None:
            lev['Interest Coverage'] = icr
        ratios['Leverage'] = pd.DataFrame(lev).T if lev else pd.DataFrame()

        return {k: v for k, v in ratios.items() if not v.empty}

    def _trends(self, df: pd.DataFrame) -> Dict[str, Dict]:
        num = df.select_dtypes(include=[np.number])
        if len(num.columns) < 2:
            return {}
        trends = {}
        for idx in num.index:
            s = num.loc[idx]
            if isinstance(s, pd.DataFrame):
                s = s.iloc[0]
            s = s.dropna()
            if len(s) < 3:
                continue
            x = np.arange(len(s))
            coef = np.polyfit(x, s.values, 1)
            slope = float(coef[0])
            # CAGR
            cagr = 0
            if s.iloc[0] > 0 and s.iloc[-1] > 0 and len(s) > 1:
                cagr = ((s.iloc[-1] / s.iloc[0]) ** (1 / (len(s) - 1)) - 1) * 100
            trends[str(idx)] = {
                'direction': 'increasing' if slope > 0 else 'decreasing',
                'cagr': round(cagr, 2),
                'volatility': round(float(s.pct_change().std() * 100), 2)}
        return trends

    def _anomalies(self, df: pd.DataFrame) -> Dict:
        anomalies = {'value': [], 'trend': []}
        num = df.select_dtypes(include=[np.number])
        for idx in num.index:
            s = num.loc[idx]
            if isinstance(s, pd.DataFrame):
                s = s.iloc[0]
            s = s.dropna()
            if len(s) < 4:
                continue
            Q1, Q3 = s.quantile(0.25), s.quantile(0.75)
            IQR = Q3 - Q1
            if IQR > 0:
                outliers = s[(s < Q1 - 3 * IQR) | (s > Q3 + 3 * IQR)]
                for year, val in outliers.items():
                    anomalies['value'].append({
                        'metric': str(idx), 'year': str(year), 'value': float(val)})
        return anomalies

    def _quality(self, df: pd.DataFrame) -> float:
        num = df.select_dtypes(include=[np.number])
        if num.empty:
            return 0
        completeness = num.notna().sum().sum() / num.size * 100
        return min(completeness, 100)

    def _insights(self, analysis: Dict) -> List[str]:
        insights = []
        summary = analysis.get('summary', {})
        quality = analysis.get('quality_score', 0)
        if quality >= 80:
            insights.append("✅ High data quality — analysis is reliable")
        elif quality < 50:
            insights.append("⚠️ Low data quality — consider checking mappings")
        trends = analysis.get('trends', {})
        rev_trends = [v for k, v in trends.items() if 'revenue' in k.lower()]
        if rev_trends:
            cagr = rev_trends[0].get('cagr', 0)
            if cagr > 15:
                insights.append(f"🚀 Strong revenue growth (CAGR: {cagr:.1f}%)")
            elif cagr < 0:
                insights.append(f"📉 Declining revenue (CAGR: {cagr:.1f}%)")
        return insights


# ═══════════════════════════════════════════════════════════
# SECTION 12: PENMAN-NISSIM ANALYZER (COMPLETE REWRITE)
# ═══════════════════════════════════════════════════════════

class PenmanNissimAnalyzer:
    """
    Penman-Nissim financial analysis framework.
    
    Academic basis:
    - Nissim & Penman (2001): Ratio Analysis and Equity Valuation
    - Penman (2013): Financial Statement Analysis and Security Valuation, 5th ed.
    
    Core decomposition: ROE = RNOA + (FLEV × Spread)
    where Spread = RNOA - NBC
    
    Key innovation in this implementation:
    - Data restructured ONCE, cached for all calculations
    - Statement-type validation prevents cross-contamination
    - Automatic Total Liabilities derivation if missing
    - Proper handling of debt-free (cash-rich) companies
    """

    def __init__(self, df: pd.DataFrame, mappings: Dict[str, str]):
        self.log = Log.get('PenmanNissim')
        self._raw = df
        self._mappings = mappings
        self._inv_map = {v: k for k, v in mappings.items()}  # target → source
        self._data = self._restructure(df)
        self._ref_bs: Optional[pd.DataFrame] = None
        self._ref_is: Optional[pd.DataFrame] = None

    # ── Data Access ──

    def _get(self, target: str, default_zero: bool = False) -> pd.Series:
        """Get a series for a target metric. Statement-type validated."""
        source = self._inv_map.get(target)
        if source and source in self._data.index:
            s = self._data.loc[source]
            if isinstance(s, pd.DataFrame):
                s = s.iloc[0]
            return s.fillna(0) if default_zero else s

        # Derived metric: Total Liabilities
        if target == 'Total Liabilities':
            try:
                return self._get('Total Assets') - self._get('Total Equity')
            except Exception:
                pass

        if default_zero:
            return pd.Series(0, index=self._data.columns, dtype=float)
        raise ValueError(f"Metric '{target}' not found in mappings")

    def _has(self, target: str) -> bool:
        return target in self._inv_map or target == 'Total Liabilities'

    # ── Restructuring ──

    def _restructure(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalise column structure to YYYYMM format. Run once."""
        year_map = YearDetector.detect_columns(df)
        if not year_map:
            self.log.warning("No year columns detected, using raw columns")
            return df

        years = sorted(year_map.keys())
        out = pd.DataFrame(index=df.index, columns=years, dtype=float)

        for year, src_cols in year_map.items():
            for idx in df.index:
                for col in src_cols:
                    try:
                        val = df.loc[idx, col]
                        if pd.notna(val):
                            num = float(str(val).replace(',', '').replace('(', '-').replace(')', '').strip())
                            if pd.isna(out.loc[idx, year]) or out.loc[idx, year] == 0:
                                out.loc[idx, year] = num
                    except (ValueError, TypeError):
                        continue

        self.log.info(f"Restructured: {out.shape}, years={years}")
        return out

    # ── Balance Sheet Reformulation ──

    def reformulate_balance_sheet(self) -> pd.DataFrame:
        """Reformulate BS into Operating vs Financial assets/liabilities."""
        if self._ref_bs is not None:
            return self._ref_bs

        self.log.info("=== Reformulating Balance Sheet ===")
        cols = self._data.columns
        ref = pd.DataFrame(index=cols, dtype=float)

        # Core items
        total_assets = self._get('Total Assets')
        total_equity = self._get('Total Equity')
        total_liab = self._get('Total Liabilities', default_zero=True)

        # Financial Assets
        cash = self._get('Cash and Cash Equivalents', True)
        curr_inv = self._get('Short-term Investments', True) if self._has('Short-term Investments') else pd.Series(0, index=cols)
        lt_inv = pd.Series(0, index=cols)  # Can be extended

        financial_assets = cash + curr_inv + lt_inv

        # Financial Liabilities (Debt)
        st_debt = self._get('Short-term Debt', True)
        lt_debt = self._get('Long-term Debt', True)
        financial_liab = st_debt + lt_debt

        # Net Financial Assets/Obligations
        nfa = financial_assets - financial_liab

        # Operating (residual)
        operating_assets = total_assets - financial_assets
        operating_liab = (total_liab - financial_liab).clip(lower=0)
        noa = operating_assets - operating_liab

        # Store
        ref['Total Assets'] = total_assets
        ref['Operating Assets'] = operating_assets
        ref['Financial Assets'] = financial_assets
        ref['Total Liabilities'] = total_liab
        ref['Operating Liabilities'] = operating_liab
        ref['Financial Liabilities'] = financial_liab
        ref['Net Operating Assets'] = noa
        ref['Net Financial Assets'] = nfa
        ref['Common Equity'] = total_equity
        ref['Total Debt'] = financial_liab
        ref['Cash and Equivalents'] = cash

        # Validate: NOA + NFA ≈ CE
        check = (noa + nfa - total_equity).abs()
        if check.max() > 1:
            self.log.warning(f"BS check: max imbalance = {check.max():.0f}")

        self.log.info(f"BS reformulation complete. NOA range: {noa.min():.0f} – {noa.max():.0f}")
        self._ref_bs = ref.T  # Rows = items, Cols = years
        return self._ref_bs

    # ── Income Statement Reformulation ──

    def reformulate_income_statement(self) -> pd.DataFrame:
        """Reformulate IS with tax allocation."""
        if self._ref_is is not None:
            return self._ref_is

        self.log.info("=== Reformulating Income Statement ===")
        cols = self._data.columns
        ref = pd.DataFrame(index=cols, dtype=float)

        revenue = self._get('Revenue')
        pbt = self._get('Income Before Tax', True)
        tax = self._get('Tax Expense', True)
        finance_cost = self._get('Interest Expense', True)
        other_income = self._get('Other Income', True) if self._has('Other Income') else pd.Series(0, index=cols)
        net_income = self._get('Net Income')

        # EBIT = PBT + Finance Cost
        ebit = pbt + finance_cost

        # Operating Income = EBIT - Other Income (conservative: treat Other Income as non-operating)
        operating_income_bt = ebit - other_income

        # Effective tax rate
        eff_tax = pd.Series(0.25, index=cols)  # Default 25%
        for year in cols:
            if pd.notna(pbt[year]) and pbt[year] > 0 and pd.notna(tax[year]):
                rate = tax[year] / pbt[year]
                if 0.05 <= rate <= 0.5:
                    eff_tax[year] = rate

        # Tax allocation
        tax_on_operating = operating_income_bt * eff_tax
        tax_on_financial = (other_income - finance_cost) * eff_tax
        # Reconciliation adjustment
        tax_allocated = tax_on_operating + tax_on_financial
        for year in cols:
            if pd.notna(tax[year]) and abs(tax_allocated[year]) > EPS:
                adj = tax[year] - tax_allocated[year]
                tax_on_operating[year] += adj

        # After-tax components
        oi_at = operating_income_bt - tax_on_operating
        nfi_at = (other_income - finance_cost) - tax_on_financial

        ref['Revenue'] = revenue
        ref['EBIT'] = ebit
        ref['Operating Income Before Tax'] = operating_income_bt
        ref['Tax on Operating Income'] = tax_on_operating
        ref['Operating Income After Tax'] = oi_at
        ref['Net Financial Income Before Tax'] = other_income - finance_cost
        ref['Net Financial Income After Tax'] = nfi_at
        ref['Net Financial Expense After Tax'] = -nfi_at
        ref['Interest Expense'] = finance_cost
        ref['Other Income'] = other_income
        ref['Effective Tax Rate'] = eff_tax
        ref['Net Income (Reported)'] = net_income
        ref['Net Income (Calculated)'] = oi_at + nfi_at

        # Gross Profit
        if self._has('Cost of Goods Sold'):
            cogs = self._get('Cost of Goods Sold', True)
            ref['Gross Profit'] = revenue - cogs

        # EBITDA
        if self._has('Depreciation'):
            dep = self._get('Depreciation', True)
            ref['EBITDA'] = operating_income_bt + dep
            ref['Depreciation'] = dep

        self.log.info("IS reformulation complete")
        self._ref_is = ref.T
        return self._ref_is

    # ── Ratio Calculations ──

    def calculate_ratios(self) -> pd.DataFrame:
        """Calculate all Penman-Nissim ratios."""
        bs = self.reformulate_balance_sheet()
        is_ = self.reformulate_income_statement()
        years = bs.columns

        ratios = pd.DataFrame(index=years, dtype=float)

        def _avg(series_name: str, source: pd.DataFrame) -> pd.Series:
            """Compute average (current + prior year) / 2."""
            s = source.loc[series_name] if series_name in source.index else pd.Series(0, index=years)
            avg = pd.Series(index=years, dtype=float)
            for i, y in enumerate(years):
                avg[y] = s[y] if i == 0 else (s[years[i - 1]] + s[y]) / 2
            return avg

        # === RNOA ===
        nopat = is_.loc['Operating Income After Tax'] if 'Operating Income After Tax' in is_.index else pd.Series(np.nan, index=years)
        avg_noa = _avg('Net Operating Assets', bs)

        rnoa = pd.Series(np.nan, index=years)
        for y in years:
            if pd.notna(avg_noa[y]) and abs(avg_noa[y]) > 10 and pd.notna(nopat[y]):
                rnoa[y] = (nopat[y] / avg_noa[y]) * 100
        ratios['Return on Net Operating Assets (RNOA) %'] = rnoa

        # === OPM & NOAT ===
        revenue = is_.loc['Revenue'] if 'Revenue' in is_.index else pd.Series(np.nan, index=years)
        opm = pd.Series(np.nan, index=years)
        noat = pd.Series(np.nan, index=years)
        for y in years:
            if pd.notna(revenue[y]) and revenue[y] > 0 and pd.notna(nopat[y]):
                opm[y] = (nopat[y] / revenue[y]) * 100
            if pd.notna(avg_noa[y]) and abs(avg_noa[y]) > 10 and pd.notna(revenue[y]):
                noat[y] = revenue[y] / avg_noa[y]
        ratios['Operating Profit Margin (OPM) %'] = opm
        ratios['Net Operating Asset Turnover (NOAT)'] = noat

        # === FLEV ===
        nfa = bs.loc['Net Financial Assets'] if 'Net Financial Assets' in bs.index else pd.Series(0, index=years)
        avg_ce = _avg('Common Equity', bs)

        flev = pd.Series(np.nan, index=years)
        for y in years:
            if pd.notna(avg_ce[y]) and abs(avg_ce[y]) > 10 and pd.notna(nfa[y]):
                flev[y] = -nfa[y] / avg_ce[y]
        ratios['Financial Leverage (FLEV)'] = flev

        # === NBC ===
        nfe_at = is_.loc['Net Financial Expense After Tax'] if 'Net Financial Expense After Tax' in is_.index else pd.Series(0, index=years)
        avg_nfo = -_avg('Net Financial Assets', bs)  # NFO = -NFA

        nbc = pd.Series(0.0, index=years)
        total_debt = bs.loc['Total Debt'] if 'Total Debt' in bs.index else pd.Series(0, index=years)

        for y in years:
            if pd.notna(avg_nfo[y]) and abs(avg_nfo[y]) > 10 and pd.notna(nfe_at[y]):
                nbc[y] = np.clip((nfe_at[y] / avg_nfo[y]) * 100, -15, 25)
            elif total_debt[y] <= 10:
                nbc[y] = 0  # Debt-free
        ratios['Net Borrowing Cost (NBC) %'] = nbc

        # === Spread & Leverage Spread ===
        spread = rnoa - nbc
        ratios['Spread %'] = spread
        ratios['Leverage Spread %'] = spread

        # === ROE ===
        net_income = is_.loc['Net Income (Reported)'] if 'Net Income (Reported)' in is_.index else pd.Series(np.nan, index=years)
        roe = pd.Series(np.nan, index=years)
        for y in years:
            if pd.notna(avg_ce[y]) and abs(avg_ce[y]) > 10 and pd.notna(net_income[y]):
                roe[y] = (net_income[y] / avg_ce[y]) * 100
        ratios['Return on Equity (ROE) %'] = roe
        ratios['ROE (Calculated) %'] = rnoa + flev * spread

        # === ROA ===
        avg_ta = _avg('Total Assets', bs)
        roa = pd.Series(np.nan, index=years)
        for y in years:
            if pd.notna(avg_ta[y]) and avg_ta[y] > 0 and pd.notna(net_income[y]):
                roa[y] = (net_income[y] / avg_ta[y]) * 100
        ratios['Return on Assets (ROA) %'] = roa

        # === Margins ===
        if 'Gross Profit' in is_.index:
            gp = is_.loc['Gross Profit']
            gpm = pd.Series(np.nan, index=years)
            for y in years:
                if pd.notna(revenue[y]) and revenue[y] > 0:
                    gpm[y] = (gp[y] / revenue[y]) * 100
            ratios['Gross Profit Margin %'] = gpm

        npm = pd.Series(np.nan, index=years)
        for y in years:
            if pd.notna(revenue[y]) and revenue[y] > 0 and pd.notna(net_income[y]):
                npm[y] = (net_income[y] / revenue[y]) * 100
        ratios['Net Profit Margin %'] = npm

        if 'EBITDA' in is_.index:
            ebitda = is_.loc['EBITDA']
            ebitda_m = pd.Series(np.nan, index=years)
            for y in years:
                if pd.notna(revenue[y]) and revenue[y] > 0:
                    ebitda_m[y] = (ebitda[y] / revenue[y]) * 100
            ratios['EBITDA Margin %'] = ebitda_m

        # === Growth ===
        ratios['Revenue Growth %'] = revenue.pct_change() * 100
        ratios['NOA Growth %'] = bs.loc['Net Operating Assets'].pct_change() * 100 if 'Net Operating Assets' in bs.index else np.nan

        # === Liquidity ===
        ca = self._get('Current Assets', True)
        cl = self._get('Current Liabilities', True)
        ratios['Current Ratio'] = pd.Series(
            [ca[y] / cl[y] if pd.notna(cl[y]) and cl[y] > 0 else np.nan for y in years], index=years)

        # === Interest Coverage ===
        ebit_vals = is_.loc['EBIT'] if 'EBIT' in is_.index else pd.Series(np.nan, index=years)
        fin_exp = is_.loc['Interest Expense'] if 'Interest Expense' in is_.index else pd.Series(0, index=years)
        icr = pd.Series(np.nan, index=years)
        for y in years:
            if pd.notna(ebit_vals[y]):
                if pd.notna(fin_exp[y]) and fin_exp[y] > 0.01:
                    icr[y] = min(ebit_vals[y] / fin_exp[y], 999)
                elif total_debt[y] <= 10 and ebit_vals[y] > 0:
                    icr[y] = 999  # Debt-free
        ratios['Interest Coverage'] = icr

        # === Debt Ratios ===
        ce = bs.loc['Common Equity'] if 'Common Equity' in bs.index else pd.Series(np.nan, index=years)
        ratios['Debt to Equity'] = pd.Series(
            [total_debt[y] / ce[y] if pd.notna(ce[y]) and ce[y] > 0 else np.nan for y in years], index=years)

        # Transpose: ratios as rows, years as columns
        result = ratios.T
        self.log.info(f"Calculated {len(result)} ratios across {len(years)} years")
        return result

    # ── Free Cash Flow ──

    def calculate_fcf(self) -> pd.DataFrame:
        """Calculate free cash flow metrics."""
        cols = self._data.columns
        fcf = pd.DataFrame(index=cols, dtype=float)

        ocf = self._get('Operating Cash Flow', True)
        capex = self._get('Capital Expenditure', True).abs()

        fcf['Operating Cash Flow'] = ocf
        fcf['Capital Expenditure'] = capex
        fcf['Free Cash Flow to Firm'] = ocf - capex

        bs = self.reformulate_balance_sheet()
        if 'Total Assets' in bs.index:
            ta = bs.loc['Total Assets']
            fcf['FCF Yield %'] = ((ocf - capex) / ta.replace(0, np.nan)) * 100

        return fcf.T

    # ── Value Drivers ──

    def calculate_value_drivers(self) -> pd.DataFrame:
        """Calculate key value drivers."""
        cols = self._data.columns
        drivers = pd.DataFrame(index=cols, dtype=float)

        rev = self._get('Revenue')
        drivers['Revenue'] = rev
        drivers['Revenue Growth %'] = rev.pct_change() * 100

        is_ = self.reformulate_income_statement()
        if 'Operating Income After Tax' in is_.index:
            nopat = is_.loc['Operating Income After Tax']
            drivers['NOPAT'] = nopat
            drivers['NOPAT Margin %'] = (nopat / rev.replace(0, np.nan)) * 100

        bs = self.reformulate_balance_sheet()
        if 'Net Operating Assets' in bs.index:
            noa = bs.loc['Net Operating Assets']
            drivers['NOA'] = noa
            drivers['NOA Growth %'] = noa.pct_change() * 100

        return drivers.T

    # ── Complete Analysis ──

    def calculate_all(self) -> Dict[str, Any]:
        """Run all Penman-Nissim calculations."""
        try:
            return {
                'reformulated_balance_sheet': self.reformulate_balance_sheet(),
                'reformulated_income_statement': self.reformulate_income_statement(),
                'ratios': self.calculate_ratios(),
                'free_cash_flow': self.calculate_fcf(),
                'value_drivers': self.calculate_value_drivers(),
                'quality_score': self._mapping_quality(),
            }
        except Exception as e:
            self.log.error(f"PN analysis failed: {e}", exc_info=True)
            return {'error': str(e)}

    def _mapping_quality(self) -> float:
        critical = ['Total Assets', 'Total Equity', 'Revenue', 'Net Income',
                    'Operating Income', 'Interest Expense', 'Tax Expense']
        found = sum(1 for m in critical if self._has(m))
        return (found / len(critical)) * 100

    def generate_insights(self) -> List[str]:
        """Generate PN-specific insights."""
        insights = []
        try:
            ratios = self.calculate_ratios()
            if ratios.empty:
                return ["⚠️ No ratios calculated"]

            latest = ratios.columns[-1]

            # RNOA
            if 'Return on Net Operating Assets (RNOA) %' in ratios.index:
                v = ratios.loc['Return on Net Operating Assets (RNOA) %', latest]
                if pd.notna(v):
                    label = "🚀 Excellent" if v > 20 else "✅ Good" if v > 10 else "⚠️ Weak"
                    insights.append(f"{label} RNOA: {v:.1f}%")

            # Spread
            if 'Spread %' in ratios.index:
                v = ratios.loc['Spread %', latest]
                if pd.notna(v):
                    if v > 0:
                        insights.append(f"✅ Positive spread ({v:.1f}%) — leverage creates value")
                    else:
                        insights.append(f"❌ Negative spread ({v:.1f}%) — leverage destroys value")

            # FLEV
            if 'Financial Leverage (FLEV)' in ratios.index:
                v = ratios.loc['Financial Leverage (FLEV)', latest]
                if pd.notna(v) and v < 0:
                    insights.append(f"💰 Net cash position (FLEV: {v:.2f})")
                elif pd.notna(v) and v > 2:
                    insights.append(f"⚠️ High leverage (FLEV: {v:.2f})")

            # ROE decomposition
            if all(r in ratios.index for r in ['Return on Equity (ROE) %', 'Return on Net Operating Assets (RNOA) %']):
                roe_v = ratios.loc['Return on Equity (ROE) %', latest]
                rnoa_v = ratios.loc['Return on Net Operating Assets (RNOA) %', latest]
                if pd.notna(roe_v) and pd.notna(rnoa_v):
                    lev_effect = roe_v - rnoa_v
                    if abs(rnoa_v) > abs(lev_effect):
                        insights.append("💡 ROE driven primarily by operations")
                    else:
                        insights.append("💡 ROE significantly influenced by leverage")

        except Exception as e:
            insights.append(f"⚠️ Insight generation error: {e}")
        return insights


# ═══════════════════════════════════════════════════════════
# SECTION 13: ML FORECASTING
# ═══════════════════════════════════════════════════════════

class Forecaster:
    """Simple ML-based financial forecasting."""

    @staticmethod
    def forecast(series: pd.Series, periods: int = 3, model: str = 'auto') -> Dict:
        """Forecast a single time series."""
        s = series.dropna()
        if len(s) < 3:
            return {'error': 'Insufficient data'}

        X = np.arange(len(s)).reshape(-1, 1)
        y = s.values

        # Auto-select best model
        models = {
            'linear': make_pipeline(LinearRegression()),
            'polynomial': make_pipeline(PolynomialFeatures(2), LinearRegression()),
        }

        if model != 'auto':
            best_model = models.get(model, models['linear'])
            best_model.fit(X, y)
        else:
            best_model, best_mse = None, float('inf')
            split = max(1, len(s) // 5)
            for name, mdl in models.items():
                mdl.fit(X[:-split], y[:-split])
                pred = mdl.predict(X[-split:])
                mse = np.mean((y[-split:] - pred) ** 2)
                if mse < best_mse:
                    best_mse = mse
                    best_model = mdl
            best_model.fit(X, y)

        # Generate predictions
        future_X = np.arange(len(s), len(s) + periods).reshape(-1, 1)
        preds = best_model.predict(future_X)

        # Confidence intervals
        residuals = y - best_model.predict(X)
        std = np.std(residuals)
        z = 1.96  # 95% CI

        # Future period labels
        try:
            last_year = int(str(s.index[-1])[:4])
            future_labels = [str(last_year + i + 1) for i in range(periods)]
        except (ValueError, IndexError):
            future_labels = [f"T+{i + 1}" for i in range(periods)]

        return {
            'periods': future_labels,
            'values': preds.tolist(),
            'lower': (preds - z * std).tolist(),
            'upper': (preds + z * std).tolist(),
            'accuracy': {
                'rmse': float(np.sqrt(np.mean(residuals ** 2))),
                'mae': float(np.mean(np.abs(residuals))),
            }
        }

    @staticmethod
    def forecast_multiple(df: pd.DataFrame, periods: int = 3,
                         metrics: Optional[List[str]] = None) -> Dict:
        """Forecast multiple metrics."""
        if metrics is None:
            num = df.select_dtypes(include=[np.number])
            metrics = num.index[:10].tolist()

        results = {}
        for metric in metrics:
            if metric in df.index:
                s = df.loc[metric]
                if isinstance(s, pd.DataFrame):
                    s = s.iloc[0]
                results[metric] = Forecaster.forecast(s.dropna(), periods)

        return results


# ═══════════════════════════════════════════════════════════
# SECTION 14: NUMBER FORMATTING & EXPORT
# ═══════════════════════════════════════════════════════════

def fmt_indian(val: float) -> str:
    if pd.isna(val):
        return "-"
    a, sign = abs(val), "-" if val < 0 else ""
    if a >= 1e7:
        return f"{sign}₹{a / 1e7:.2f} Cr"
    if a >= 1e5:
        return f"{sign}₹{a / 1e5:.2f} L"
    if a >= 1e3:
        return f"{sign}₹{a / 1e3:.1f} K"
    return f"{sign}₹{a:.0f}"


def fmt_intl(val: float) -> str:
    if pd.isna(val):
        return "-"
    a, sign = abs(val), "-" if val < 0 else ""
    if a >= 1e9:
        return f"{sign}${a / 1e9:.2f}B"
    if a >= 1e6:
        return f"{sign}${a / 1e6:.2f}M"
    if a >= 1e3:
        return f"{sign}${a / 1e3:.1f}K"
    return f"{sign}${a:.0f}"


@lru_cache(maxsize=2)
def get_formatter(fmt: str) -> Callable:
    return fmt_indian if fmt == 'Indian' else fmt_intl


class ExportManager:
    """Export analysis to various formats."""

    @staticmethod
    def to_excel(analysis: Dict, company: str = 'Analysis') -> bytes:
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine='xlsxwriter') as w:
            if 'summary' in analysis:
                pd.DataFrame([analysis['summary']]).to_excel(w, 'Summary', index=False)
            for cat, df in analysis.get('ratios', {}).items():
                if isinstance(df, pd.DataFrame) and not df.empty:
                    df.to_excel(w, cat[:31])
            if 'insights' in analysis:
                pd.DataFrame({'Insights': analysis['insights']}).to_excel(w, 'Insights', index=False)
        buf.seek(0)
        return buf.read()

    @staticmethod
    def to_markdown(analysis: Dict, company: str = 'Analysis') -> str:
        lines = [
            f"# {company} Financial Analysis Report",
            f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "\n---\n",
            "## Summary\n",
        ]
        s = analysis.get('summary', {})
        lines.extend([f"- **{k}**: {v}" for k, v in s.items()])
        if 'insights' in analysis:
            lines.append("\n## Insights\n")
            lines.extend([f"- {i}" for i in analysis['insights']])
        return '\n'.join(lines)


# ═══════════════════════════════════════════════════════════
# SECTION 15: SAMPLE DATA
# ═══════════════════════════════════════════════════════════

class SampleData:
    """Generate sample financial datasets for demo purposes."""

    @staticmethod
    def indian_tech() -> Tuple[pd.DataFrame, str]:
        years = ['201903', '202003', '202103', '202203', '202303']
        data = {
            'BalanceSheet::Total Assets': [45000, 52000, 61000, 72000, 85000],
            'BalanceSheet::Current Assets': [28000, 32000, 38000, 45000, 53000],
            'BalanceSheet::Cash and Cash Equivalents': [12000, 14000, 17000, 21000, 25000],
            'BalanceSheet::Inventories': [2000, 2300, 2700, 3200, 3800],
            'BalanceSheet::Trade Receivables': [8000, 9200, 10800, 12700, 15000],
            'BalanceSheet::Fixed Assets': [10000, 12000, 14000, 16500, 19500],
            'BalanceSheet::Total Equity': [27000, 32000, 38500, 46500, 56000],
            'BalanceSheet::Share Capital': [1000, 1000, 1000, 1000, 1000],
            'BalanceSheet::Other Equity': [26000, 31000, 37500, 45500, 55000],
            'BalanceSheet::Total Current Liabilities': [10000, 11000, 12500, 14000, 16000],
            'BalanceSheet::Trade Payables': [4000, 4400, 4900, 5500, 6200],
            'BalanceSheet::Short Term Borrowings': [3000, 3300, 3700, 4200, 4800],
            'BalanceSheet::Long Term Borrowings': [6000, 6600, 7300, 8200, 9200],
            'ProfitLoss::Revenue From Operations': [35000, 38000, 45000, 54000, 65000],
            'ProfitLoss::Cost of Materials Consumed': [21000, 22000, 25200, 29700, 35100],
            'ProfitLoss::Employee Benefit Expenses': [4000, 4400, 5150, 6075, 7150],
            'ProfitLoss::Other Expenses': [4000, 4400, 5150, 6075, 7150],
            'ProfitLoss::Profit Before Exceptional Items and Tax': [6000, 7200, 9500, 12150, 15600],
            'ProfitLoss::Finance Cost': [800, 880, 970, 1090, 1220],
            'ProfitLoss::Other Income': [400, 460, 530, 600, 700],
            'ProfitLoss::Profit Before Tax': [5200, 6320, 8530, 11060, 14380],
            'ProfitLoss::Tax Expense': [1560, 1896, 2559, 3318, 4314],
            'ProfitLoss::Profit After Tax': [3640, 4424, 5971, 7742, 10066],
            'ProfitLoss::Depreciation and Amortisation Expenses': [1500, 1800, 2100, 2500, 3000],
            'CashFlow::Net Cash from Operating Activities': [5500, 6600, 8800, 11000, 14000],
            'CashFlow::Purchase of Fixed Assets': [2800, 3200, 3800, 4500, 5300],
        }
        df = pd.DataFrame(data, columns=years).T
        df.columns = years
        # Fix: data dict keys are index, years are columns
        df2 = pd.DataFrame(data, index=list(data.keys()))
        df2.columns = years
        return df2, "TechCorp India Ltd."


# ═══════════════════════════════════════════════════════════
# SECTION 16: MAIN APPLICATION CLASS
# ═══════════════════════════════════════════════════════════

class App:
    """Main Streamlit application — clean, minimal, no redundancy."""

    def __init__(self):
        self._init_state()
        Cfg.from_session()
        self.log = Log.get('App')
        self.parser = CapitalineParser()
        self.cleaner = DataCleaner()
        self.analyzer = FinancialAnalyzer()
        self.mapper = MetricMapper()
        self.compressor = CompressedFileHandler()

        # Lazy init mapper
        if not st.session_state.get('mapper_initialized'):
            self.mapper.initialize(Cfg.KAGGLE_URL)
            st.session_state.mapper_initialized = True

    def _init_state(self):
        defaults = {
            'data': None, 'data_hash': None, 'company': None,
            'mappings': None, 'pn_mappings': None, 'pn_results': None,
            'number_format': 'Indian', 'debug_mode': False,
            'kaggle_api_url': '', 'mapper_initialized': False,
            'forecast_results': None,
        }
        for k, v in defaults.items():
            if k not in st.session_state:
                st.session_state[k] = v

    def run(self):
        self._css()
        self._header()
        self._sidebar()
        self._main()

    def _css(self):
        st.markdown("""<style>
        .main-title {font-size:2.5rem;font-weight:800;
            background:linear-gradient(135deg,#667eea,#764ba2);
            -webkit-background-clip:text;-webkit-text-fill-color:transparent;
            text-align:center;padding:1rem 0}
        .stMetric {background:#fff;padding:0.8rem;border-radius:0.5rem;
            box-shadow:0 2px 4px rgba(0,0,0,0.1)}
        </style>""", unsafe_allow_html=True)

    def _header(self):
        st.markdown('<h1 class="main-title">💹 Elite Financial Analytics v6.0</h1>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        status = self.mapper.get_status()
        c1.metric("Version", Cfg.VERSION)
        c2.metric("AI", "Kaggle GPU" if status['kaggle_ok'] else "Local" if status['local_ok'] else "Fuzzy")
        c3.metric("Cache Hit", f"{status['cache_hit_rate']:.0f}%")
        c4.metric("Format", Cfg.NUMBER_FORMAT)

    def _sidebar(self):
        st.sidebar.title("⚙️ Configuration")

        # Kaggle
        with st.sidebar.expander("🖥️ Kaggle GPU"):
            url = st.text_input("Ngrok URL", st.session_state.kaggle_api_url,
                              placeholder="https://xxxx.ngrok-free.app")
            if st.button("Test Connection"):
                if url:
                    st.session_state.kaggle_api_url = url
                    Cfg.KAGGLE_URL = url
                    self.mapper.initialize(url)
                    if self.mapper.get_status()['kaggle_ok']:
                        st.success("✅ Connected!")
                    else:
                        st.error("❌ Failed")

        # Data input
        st.sidebar.header("📥 Data Input")
        method = st.sidebar.radio("Method", ["Upload", "Sample Data"])

        if method == "Upload":
            files = st.sidebar.file_uploader(
                "Upload Financial Statements", type=Cfg.ALLOWED_TYPES,
                accept_multiple_files=True)
            if files and st.sidebar.button("Process", type="primary"):
                self._process_files(files)
        else:
            if st.sidebar.button("Load Indian Tech Sample", type="primary"):
                df, name = SampleData.indian_tech()
                st.session_state.data = df
                st.session_state.company = name
                st.sidebar.success(f"Loaded: {name}")

        # Settings
        st.sidebar.header("⚙️ Settings")
        fmt = st.sidebar.radio("Number Format", ["Indian", "International"])
        st.session_state.number_format = fmt
        Cfg.NUMBER_FORMAT = fmt
        Cfg.DEBUG = st.sidebar.checkbox("Debug Mode", Cfg.DEBUG)

    def _process_files(self, files):
        """Process uploaded files with progress tracking."""
        all_dfs = []
        prog = st.progress(0)

        for i, f in enumerate(files):
            prog.progress((i + 1) / len(files))
            try:
                if f.name.lower().endswith(('.zip', '.7z')):
                    for name, content in self.compressor.extract(f):
                        buf = io.BytesIO(content); buf.name = name
                        df = self.parser.parse(buf)
                        if df is not None and not df.empty:
                            df, _ = self.cleaner.clean(df)
                            all_dfs.append(df)
                else:
                    val = self.cleaner.validate_file(f)
                    if not val.valid:
                        st.error(f"❌ {f.name}: {val.errors[0]}")
                        continue
                    df = self.parser.parse(f)
                    if df is not None and not df.empty:
                        df, _ = self.cleaner.clean(df)
                        all_dfs.append(df)
            except Exception as e:
                st.error(f"Error: {f.name}: {e}")

        prog.empty()
        self.compressor.cleanup()

        if all_dfs:
            combined = pd.concat(all_dfs, axis=0)
            combined = combined[~combined.index.duplicated(keep='first')]
            st.session_state.data = combined
            st.session_state.mappings = None
            st.session_state.pn_mappings = None
            st.success(f"✅ Processed {len(files)} file(s): {combined.shape[0]} metrics, {combined.shape[1]} periods")
        else:
            st.error("No valid data found")

    def _main(self):
        data = st.session_state.data
        if data is None:
            self._welcome()
            return

        tabs = st.tabs(["📊 Overview", "📈 Ratios", "📉 Trends",
                        "🎯 Penman-Nissim", "🔍 Explorer", "📄 Export"])
        with tabs[0]:
            self._tab_overview(data)
        with tabs[1]:
            self._tab_ratios(data)
        with tabs[2]:
            self._tab_trends(data)
        with tabs[3]:
            self._tab_pn(data)
        with tabs[4]:
            self._tab_explorer(data)
        with tabs[5]:
            self._tab_export(data)

    def _welcome(self):
        st.header("Welcome to Elite Financial Analytics v6.0")
        c1, c2, c3 = st.columns(3)
        c1.info("### 📊 Analytics\n- Penman-Nissim Framework\n- 27+ Financial Ratios\n- ML Forecasting")
        c2.success("### 🤖 AI Features\n- Auto metric mapping\n- Kaggle GPU support\n- Pattern recognition")
        c3.warning("### 📦 Data Support\n- Capitaline exports\n- Excel/CSV/HTML\n- ZIP/7Z archives")

    # ── Tab: Overview ──

    def _tab_overview(self, data: pd.DataFrame):
        st.header("Financial Overview")
        analysis = self.analyzer.analyze(data)
        s = analysis['summary']

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Metrics", s['total_metrics'])
        c2.metric("Years", s['years_covered'])
        c3.metric("Completeness", f"{s['completeness']:.1f}%")
        c4.metric("Quality", f"{analysis['quality_score']:.0f}%")

        st.subheader("Key Insights")
        for insight in analysis.get('insights', []):
            if '✅' in insight or '🚀' in insight:
                st.success(insight)
            elif '⚠️' in insight:
                st.warning(insight)
            else:
                st.info(insight)

    # ── Tab: Ratios ──

    def _tab_ratios(self, data: pd.DataFrame):
        st.header("📈 Financial Ratios")
        if not st.session_state.mappings:
            if st.button("🤖 Auto-Map Metrics", type="primary"):
                sources = [str(m) for m in data.index]
                result = self.mapper.map_metrics(sources)
                st.session_state.mappings = result['mappings']
                st.success(f"Mapped {len(result['mappings'])} metrics ({result['method']})")
            return

        mapped = data.rename(index=st.session_state.mappings)
        analysis = self.analyzer.analyze(mapped)
        for cat, rdf in analysis.get('ratios', {}).items():
            if isinstance(rdf, pd.DataFrame) and not rdf.empty:
                st.subheader(f"{cat} Ratios")
                st.dataframe(rdf.style.format("{:.2f}", na_rep="-")
                           .background_gradient(cmap='RdYlGn', axis=1), use_container_width=True)

    # ── Tab: Trends ──

    def _tab_trends(self, data: pd.DataFrame):
        st.header("📉 Trends & Forecasting")
        analysis = self.analyzer.analyze(data)
        trends = analysis.get('trends', {})

        if trends:
            tdata = [{'Metric': k, 'Direction': v['direction'],
                     'CAGR %': v['cagr'], 'Volatility %': v['volatility']}
                    for k, v in trends.items() if isinstance(v, dict)]
            if tdata:
                st.dataframe(pd.DataFrame(tdata).style.format(
                    {'CAGR %': '{:.1f}', 'Volatility %': '{:.1f}'}, na_rep='-'),
                    use_container_width=True)

        # Forecasting
        st.subheader("🤖 ML Forecast")
        num_metrics = data.select_dtypes(include=[np.number]).index.tolist()
        selected = st.multiselect("Metrics to forecast", num_metrics, default=num_metrics[:3])
        periods = st.slider("Forecast periods", 1, 6, 3)

        if st.button("Generate Forecast", type="primary") and selected:
            with st.spinner("Training models..."):
                results = Forecaster.forecast_multiple(data, periods, selected)
                for metric, fc in results.items():
                    if 'error' in fc:
                        continue
                    st.subheader(metric)
                    actual = data.loc[metric].dropna()
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=actual.index.tolist(), y=actual.values.tolist(),
                                           mode='lines+markers', name='Actual'))
                    fig.add_trace(go.Scatter(x=fc['periods'], y=fc['values'],
                                           mode='lines+markers', name='Forecast',
                                           line=dict(dash='dash')))
                    if fc.get('lower'):
                        fig.add_trace(go.Scatter(x=fc['periods'], y=fc['lower'],
                                               mode='lines', showlegend=False, line=dict(width=0)))
                        fig.add_trace(go.Scatter(x=fc['periods'], y=fc['upper'],
                                               mode='lines', fill='tonexty',
                                               fillcolor='rgba(255,165,0,0.2)',
                                               showlegend=False, line=dict(width=0)))
                    fig.update_layout(height=350, hovermode='x unified')
                    st.plotly_chart(fig, use_container_width=True)

    # ── Tab: Penman-Nissim ──

    def _tab_pn(self, data: pd.DataFrame):
        st.header("🎯 Penman-Nissim Analysis")

        # Mapping
        if not st.session_state.pn_mappings:
            st.info("Configure metric mappings for Penman-Nissim analysis.")

            c1, c2 = st.columns(2)
            with c1:
                if st.button("🤖 Auto-Map", type="primary"):
                    sources = [str(m) for m in data.index]
                    mappings, unmapped = MappingTemplate.create_auto_mapping(sources)
                    st.session_state.pn_mappings = mappings
                    st.success(f"Mapped {len(mappings)} metrics, {len(unmapped)} unmapped")
                    st.rerun()
            with c2:
                if st.button("📋 Show Available Metrics"):
                    for stmt in ['BalanceSheet', 'ProfitLoss', 'CashFlow']:
                        items = [i for i in data.index if str(i).startswith(f"{stmt}::")]
                        if items:
                            with st.expander(f"{stmt} ({len(items)})"):
                                for item in items[:30]:
                                    st.code(item)
            return

        mappings = st.session_state.pn_mappings

        # Show mapping summary
        mapped_targets = set(mappings.values())
        critical = ['Total Assets', 'Total Equity', 'Revenue', 'Net Income', 'Operating Income']
        missing = [c for c in critical if c not in mapped_targets]
        if missing:
            st.warning(f"Missing: {', '.join(missing)}")

        # Run analysis
        with st.spinner("Running Penman-Nissim analysis..."):
            try:
                pn = PenmanNissimAnalyzer(data, mappings)
                results = pn.calculate_all()
                st.session_state.pn_results = results
            except Exception as e:
                st.error(f"Analysis failed: {e}")
                if Cfg.DEBUG:
                    st.code(traceback.format_exc())
                return

        if 'error' in results:
            st.error(results['error'])
            return

        # Sub-tabs
        pn_tabs = st.tabs(["📊 Ratios", "📈 Trends", "📑 Statements", "💰 Cash Flow", "🎯 Drivers"])

        with pn_tabs[0]:
            self._pn_ratios(results, pn)
        with pn_tabs[1]:
            self._pn_trends(results)
        with pn_tabs[2]:
            self._pn_statements(results)
        with pn_tabs[3]:
            self._pn_fcf(results)
        with pn_tabs[4]:
            self._pn_drivers(results)

    def _pn_ratios(self, results: Dict, pn: PenmanNissimAnalyzer):
        ratios = results.get('ratios')
        if ratios is None or ratios.empty:
            st.warning("No ratios calculated")
            return

        latest = ratios.columns[-1]
        c1, c2, c3, c4 = st.columns(4)

        def _safe_metric(col, label, key, fmt="{:.1f}%"):
            if key in ratios.index:
                v = ratios.loc[key, latest]
                col.metric(label, fmt.format(v) if pd.notna(v) else "N/A")
            else:
                col.metric(label, "N/A")

        _safe_metric(c1, "RNOA", 'Return on Net Operating Assets (RNOA) %')
        _safe_metric(c2, "FLEV", 'Financial Leverage (FLEV)', "{:.2f}")
        _safe_metric(c3, "NBC", 'Net Borrowing Cost (NBC) %')
        _safe_metric(c4, "Spread", 'Spread %')

        # ROE Decomposition Chart
        req = ['Return on Equity (ROE) %', 'Return on Net Operating Assets (RNOA) %',
               'Financial Leverage (FLEV)', 'Spread %']
        if all(r in ratios.index for r in req):
            st.subheader("ROE Decomposition")
            roe = ratios.loc['Return on Equity (ROE) %']
            rnoa = ratios.loc['Return on Net Operating Assets (RNOA) %']
            flev = ratios.loc['Financial Leverage (FLEV)']
            spread = ratios.loc['Spread %']
            lev_effect = flev * spread

            fig = go.Figure()
            fig.add_trace(go.Bar(x=ratios.columns, y=rnoa, name='RNOA', marker_color='royalblue'))
            fig.add_trace(go.Bar(x=ratios.columns, y=lev_effect, name='Leverage Effect', marker_color='gray'))
            fig.add_trace(go.Scatter(x=ratios.columns, y=roe, name='Total ROE',
                                    mode='lines+markers', line=dict(color='red', width=3)))
            fig.update_layout(barmode='relative', height=400, hovermode='x unified')
            st.plotly_chart(fig, use_container_width=True)

        # Full ratios table
        st.subheader("All Ratios")
        st.dataframe(ratios.style.format("{:.2f}", na_rep="-")
                    .background_gradient(cmap='RdYlGn', axis=1), use_container_width=True)

        # Insights
        insights = pn.generate_insights()
        if insights:
            st.subheader("💡 Insights")
            for i in insights:
                if '✅' in i or '🚀' in i or '💰' in i:
                    st.success(i)
                elif '⚠️' in i or '❌' in i:
                    st.warning(i)
                else:
                    st.info(i)

    def _pn_trends(self, results: Dict):
        ratios = results.get('ratios')
        if ratios is None or ratios.empty:
            return
        key_ratios = ['Return on Net Operating Assets (RNOA) %', 'Financial Leverage (FLEV)',
                     'Net Borrowing Cost (NBC) %', 'Spread %', 'Return on Equity (ROE) %']
        for ratio in key_ratios:
            if ratio in ratios.index:
                vals = ratios.loc[ratio]
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=ratios.columns, y=vals, mode='lines+markers',
                                        name=ratio, line=dict(width=3), marker=dict(size=8)))
                fig.update_layout(title=ratio, height=300)
                st.plotly_chart(fig, use_container_width=True)

    def _pn_statements(self, results: Dict):
        c1, c2 = st.columns(2)
        with c1:
            bs = results.get('reformulated_balance_sheet')
            if bs is not None and not bs.empty:
                st.subheader("Reformulated Balance Sheet")
                st.dataframe(bs.style.format("{:,.0f}", na_rep="-"), use_container_width=True)
        with c2:
            is_ = results.get('reformulated_income_statement')
            if is_ is not None and not is_.empty:
                st.subheader("Reformulated Income Statement")
                st.dataframe(is_.style.format("{:,.0f}", na_rep="-"), use_container_width=True)

    def _pn_fcf(self, results: Dict):
        fcf = results.get('free_cash_flow')
        if fcf is None or fcf.empty:
            st.info("No FCF data")
            return
        st.subheader("Free Cash Flow")
        st.dataframe(fcf.style.format("{:,.0f}", na_rep="-"), use_container_width=True)

        if 'Free Cash Flow to Firm' in fcf.index:
            fig = go.Figure()
            if 'Operating Cash Flow' in fcf.index:
                fig.add_trace(go.Bar(x=fcf.columns, y=fcf.loc['Operating Cash Flow'],
                                    name='OCF', marker_color='green'))
            if 'Capital Expenditure' in fcf.index:
                fig.add_trace(go.Bar(x=fcf.columns, y=-fcf.loc['Capital Expenditure'],
                                    name='CapEx', marker_color='red'))
            fig.add_trace(go.Scatter(x=fcf.columns, y=fcf.loc['Free Cash Flow to Firm'],
                                    name='FCFF', mode='lines+markers', line=dict(color='blue', width=3)))
            fig.update_layout(barmode='relative', height=400, title="Free Cash Flow Analysis")
            st.plotly_chart(fig, use_container_width=True)

    def _pn_drivers(self, results: Dict):
        drivers = results.get('value_drivers')
        if drivers is None or drivers.empty:
            return
        st.subheader("Value Drivers")
        st.dataframe(drivers.style.format("{:.2f}", na_rep="-")
                    .background_gradient(cmap='RdYlGn', axis=1), use_container_width=True)

    # ── Tab: Explorer ──

    def _tab_explorer(self, data: pd.DataFrame):
        st.header("🔍 Data Explorer")
        sel_metrics = st.multiselect("Metrics", data.index.tolist(), default=data.index[:10].tolist())
        if sel_metrics:
            filtered = data.loc[sel_metrics]
            st.dataframe(filtered, use_container_width=True)
            st.download_button("Download CSV", filtered.to_csv().encode(), "data.csv", "text/csv")

    # ── Tab: Export ──

    def _tab_export(self, data: pd.DataFrame):
        st.header("📄 Export")
        analysis = self.analyzer.analyze(data)
        company = st.session_state.get('company', 'Analysis')

        c1, c2 = st.columns(2)
        with c1:
            excel = ExportManager.to_excel(analysis, company)
            st.download_button("📊 Download Excel", excel, f"{company}.xlsx",
                             "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        with c2:
            md = ExportManager.to_markdown(analysis, company)
            st.download_button("📝 Download Markdown", md, f"{company}.md", "text/markdown")


# ═══════════════════════════════════════════════════════════
# SECTION 17: ENTRY POINT
# ═══════════════════════════════════════════════════════════

def main():
    try:
        st.set_page_config(
            page_title="Elite Financial Analytics v6.0",
            page_icon="💹", layout="wide",
            initial_sidebar_state="expanded")
        App().run()
    except Exception as e:
        st.error(f"🚨 Critical error: {e}")
        if st.button("🔄 Reset"):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()
        if st.checkbox("Show details"):
            st.code(traceback.format_exc())


if __name__ == "__main__":
    main()

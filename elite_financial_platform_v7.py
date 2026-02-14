# elite_financial_platform_v7.py
# Enterprise Financial Analytics Platform v7.0 — Complete Enhanced Rewrite
# All bugs fixed, missing features added, deep Penman-Nissim framework
#
# CHANGELOG from v6:
#   ✅ FIXED: SampleData.indian_tech() DataFrame construction bug
#   ✅ FIXED: Short-term Investments pattern missing (PN analyzer referenced it)
#   ✅ FIXED: _normalise_columns dropping data prematurely
#   ✅ FIXED: ExportManager not including PN results
#   ✅ FIXED: Operating Income / EBIT pattern collision
#   ✅ ADDED: 25+ missing MetricPatterns (Goodwill, Intangibles, EPS, Dividends, etc.)
#   ✅ ADDED: DuPont 3-Factor & 5-Factor Decomposition
#   ✅ ADDED: Altman Z-Score
#   ✅ ADDED: Piotroski F-Score
#   ✅ ADDED: Residual Income & Economic Value Added
#   ✅ ADDED: Sustainable Growth Rate
#   ✅ ADDED: Working Capital Analysis
#   ✅ ADDED: Manual Mapping Editor UI
#   ✅ ADDED: Company Name Input
#   ✅ ADDED: Data Preview Tab
#   ✅ ADDED: Radar/Spider Charts
#   ✅ ADDED: Waterfall Charts for ROE Decomposition
#   ✅ ADDED: Correlation Heatmap
#   ✅ ADDED: Gauge Charts
#   ✅ ADDED: Benchmark Comparison
#   ✅ ADDED: PDF Export (via HTML)
#   ✅ ADDED: JSON Export
#   ✅ ADDED: Scenario / Sensitivity Analysis
#   ✅ ADDED: Multiple Sample Datasets (Tech, Mfg, Banking, Pharma)
#   ✅ ADDED: Unit Detection (Cr/Lakhs/Thousands)
#   ✅ ADDED: Outlier Winsorization
#   ✅ ADDED: Multi-company session support
#   ✅ ADDED: Proper PN caching in session state
#   ✅ ADDED: Input sanitization using bleach
#   ✅ ADDED: Exponential Smoothing forecaster
#   ✅ ADDED: Comprehensive error boundaries per tab
#   ✅ REMOVED: Unused imports (ABC, auto)
#
# Section Map:
#   1.  Imports & Constants
#   2.  Configuration
#   3.  Logging & Performance
#   4.  Caching
#   5.  Year Detection & Parsing Utilities
#   6.  Capitaline Parser
#   7.  Data Cleaning & Validation
#   8.  Metric Pattern Matching & Mapping
#   9.  Kaggle API Client
#  10.  AI/Fuzzy Mapper
#  11.  Financial Analysis Engine (Enhanced)
#  12.  Penman-Nissim Analyzer (Enhanced)
#  13.  Advanced Scoring Models (NEW — Altman Z, Piotroski F)
#  14.  ML Forecasting (Enhanced)
#  15.  Visualization Factory (NEW)
#  16.  Number Formatting & Export (Enhanced)
#  17.  Sample Data (FIXED + Expanded)
#  18.  UI Components & Rendering
#  19.  Main Application Class (Enhanced)
#  20.  Entry Point

# ═══════════════════════════════════════════════════════════
# SECTION 1: IMPORTS & CONSTANTS
# ═══════════════════════════════════════════════════════════

import functools
import gc
import hashlib
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
import base64
from collections import defaultdict, OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
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
    import py7zr
    SEVEN_ZIP_OK = True
except ImportError:
    SEVEN_ZIP_OK = False

try:
    from sentence_transformers import SentenceTransformer
    ST_OK = True
except ImportError:
    ST_OK = False

try:
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    REQUESTS_OK = True
except ImportError:
    REQUESTS_OK = False

try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    STATSMODELS_OK = True
except ImportError:
    STATSMODELS_OK = False

EPS = 1e-10
YEAR_RE = re.compile(r'(20\d{2}|19\d{2})')
YYYYMM_RE = re.compile(r'(\d{6})')
MAX_FILE_MB = 50


# ═══════════════════════════════════════════════════════════
# SECTION 2: CONFIGURATION
# ═══════════════════════════════════════════════════════════

class Cfg:
    """Minimal, flat configuration."""

    VERSION = '7.0.0'
    DEBUG = False
    ALLOWED_TYPES = ['csv', 'html', 'htm', 'xls', 'xlsx', 'zip', '7z']

    CONFIDENCE_THRESHOLD = 0.6
    OUTLIER_STD = 3
    MIN_DATA_POINTS = 3
    WINSORIZE_LIMITS = (0.01, 0.99)

    AI_ENABLED = True
    AI_MODEL = 'all-MiniLM-L6-v2'
    KAGGLE_URL = ''
    KAGGLE_TIMEOUT = 30
    KAGGLE_RETRIES = 3
    KAGGLE_BATCH = 50

    NUMBER_FORMAT = 'Indian'
    CURRENCY_SYMBOL = '₹'
    UNIT_SCALE = 1  # 1=absolute, 1e7=Cr, 1e5=Lakhs

    # Benchmark defaults (Indian market)
    BENCHMARKS = {
        'risk_free_rate': 7.0,
        'market_return': 12.0,
        'cost_of_equity': 12.0,
        'cost_of_debt': 9.0,
        'tax_rate': 25.17,
    }

    @classmethod
    def from_session(cls):
        cls.KAGGLE_URL = st.session_state.get('kaggle_api_url', '')
        cls.NUMBER_FORMAT = st.session_state.get('number_format', 'Indian')
        cls.CURRENCY_SYMBOL = '₹' if cls.NUMBER_FORMAT == 'Indian' else '$'
        cls.DEBUG = st.session_state.get('debug_mode', False)
        cls.UNIT_SCALE = st.session_state.get('unit_scale', 1)


# ═══════════════════════════════════════════════════════════
# SECTION 3: LOGGING & PERFORMANCE
# ═══════════════════════════════════════════════════════════

class Log:
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
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        fh = RotatingFileHandler(
            cls._dir / f"{name}.log", maxBytes=5_000_000, backupCount=3)
        fh.setFormatter(fmt)
        logger.addHandler(sh)
        logger.addHandler(fh)
        cls._cache[name] = logger
        return logger


class Perf:
    _data: Dict[str, List[float]] = defaultdict(list)

    @classmethod
    @contextmanager
    def measure(cls, name: str):
        t0 = time.time()
        yield
        cls._data[name].append(time.time() - t0)

    @classmethod
    def summary(cls) -> Dict[str, Dict]:
        return {
            k: {'avg': np.mean(v), 'count': len(v), 'total': sum(v)}
            for k, v in cls._data.items() if v
        }


# ═══════════════════════════════════════════════════════════
# SECTION 4: CACHING
# ═══════════════════════════════════════════════════════════

class Cache:
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
    PATTERNS = [
        (re.compile(r'^(\d{6})$'), lambda m: m.group(1)),
        (re.compile(r'(\d{4})(\d{2})'), lambda m: m.group(0)),
        (re.compile(r'FY\s*(\d{4})'), lambda m: m.group(1) + '03'),
        (re.compile(r'Mar(?:ch)?[- ]?(\d{4})'), lambda m: m.group(1) + '03'),
        (re.compile(r'Mar(?:ch)?[- ]?(\d{2})$'),
         lambda m: '20' + m.group(1) + '03'),
        (re.compile(r'(\d{4})-(\d{2})'), lambda m: m.group(1) + '03'),
        (re.compile(r'(20\d{2}|19\d{2})'), lambda m: m.group(1) + '03'),
    ]

    @classmethod
    def extract(cls, col_name: str) -> Optional[str]:
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
        mapping = defaultdict(list)
        for col in df.columns:
            year = cls.extract(col)
            if year:
                mapping[year].append(col)
        return dict(mapping)


class UnitDetector:
    """NEW: Detect data units (Cr, Lakhs, Thousands, Absolute)."""

    @staticmethod
    def detect(df: pd.DataFrame) -> Tuple[str, float]:
        """Detect the scale of numeric values. Returns (label, multiplier)."""
        num = df.select_dtypes(include=[np.number])
        if num.empty:
            return 'Absolute', 1.0
        median_val = num.median().median()
        if pd.isna(median_val):
            return 'Absolute', 1.0
        abs_med = abs(median_val)
        if abs_med > 1e8:
            return 'Absolute (Large)', 1.0
        if abs_med > 1e5:
            return 'In Lakhs', 1e5
        if abs_med > 1e3:
            return 'In Thousands', 1e3
        if abs_med < 1:
            return 'In Crores', 1e7
        return 'Absolute', 1.0

    @staticmethod
    def detect_from_headers(df: pd.DataFrame) -> Optional[str]:
        """Detect units from column/index text clues."""
        all_text = ' '.join(str(v) for v in
                           list(df.columns) + list(df.index)[:20]).lower()
        if 'crore' in all_text or '(cr)' in all_text or 'in cr' in all_text:
            return 'Crores'
        if 'lakh' in all_text or '(l)' in all_text or 'in lakhs' in all_text:
            return 'Lakhs'
        if 'thousand' in all_text or "(000)" in all_text or "'000" in all_text:
            return 'Thousands'
        if 'million' in all_text or '(mn)' in all_text or 'in mn' in all_text:
            return 'Millions'
        return None


class StatementClassifier:
    PL_KW = {
        'revenue', 'sales', 'income', 'profit', 'loss', 'expense', 'cost',
        'ebit', 'ebitda', 'tax', 'interest', 'depreciation', 'amortisation',
        'amortization', 'dividend', 'earning', 'margin', 'turnover',
        'exceptional', 'extraordinary'
    }
    BS_KW = {
        'asset', 'liability', 'liabilities', 'equity', 'capital', 'reserve',
        'surplus', 'receivable', 'payable', 'inventory', 'inventories',
        'borrowing', 'debt', 'investment', 'property', 'plant', 'goodwill',
        'cash', 'bank', 'provision', 'debenture', 'intangible', 'tangible',
        'net block', 'net worth', 'minority'
    }
    CF_KW = {
        'cash flow', 'operating activities', 'investing activities',
        'financing activities', 'capex', 'capital expenditure',
        'purchase of fixed', 'net cash', 'free cash'
    }

    @classmethod
    def classify(cls, text: str) -> str:
        low = text.lower()
        if low.startswith('profitloss::') or low.startswith('p&l::'):
            return 'ProfitLoss'
        if low.startswith('balancesheet::') or low.startswith('bs::'):
            return 'BalanceSheet'
        if low.startswith('cashflow::') or low.startswith('cf::'):
            return 'CashFlow'
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
        blob = ' '.join(
            str(v) for v in df.values.flatten()[:300] if pd.notna(v)
        ).lower()
        scores = {
            'CashFlow': sum(1 for kw in cls.CF_KW if kw in blob) * 3,
            'ProfitLoss': sum(1 for kw in cls.PL_KW if kw in blob),
            'BalanceSheet': sum(1 for kw in cls.BS_KW if kw in blob),
        }
        return (max(scores, key=scores.get)
                if max(scores.values()) > 0 else 'Financial')

    @classmethod
    def classify_sheet(cls, sheet_name: str, df: pd.DataFrame) -> str:
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
    def __init__(self):
        self.log = Log.get('Parser')
        self.detected_unit: Optional[str] = None

    def parse(self, file) -> Optional[pd.DataFrame]:
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

    def _parse_html_tables(self, file, name: str) -> Optional[pd.DataFrame]:
        file.seek(0)
        try:
            raw_content = file.read()
            if isinstance(raw_content, bytes):
                raw_content = raw_content.decode('utf-8', errors='replace')
            # Detect units from HTML content
            self.detected_unit = UnitDetector.detect_from_headers(
                pd.DataFrame([raw_content[:2000]])
            )
            file.seek(0)
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
            if tbl.shape[0] < 3 or tbl.shape[1] < 2:
                continue
            df = self._process_html_table(tbl, i)
            if df is not None and not df.empty:
                processed.append(df)

        if not processed:
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

    def _process_html_table(
        self, tbl: pd.DataFrame, idx: int
    ) -> Optional[pd.DataFrame]:
        header_row = self._find_header_row(tbl)
        if header_row is None:
            return None
        stmt_type = StatementClassifier.classify_table(tbl)
        headers = tbl.iloc[header_row].tolist()
        data = tbl.iloc[header_row + 1:].copy()
        data.columns = headers
        metric_col = self._find_metric_column(data)
        if metric_col is not None:
            data = data.set_index(data.columns[metric_col])
        data = self._normalise_columns(data)
        for col in data.columns:
            data[col] = self._to_numeric(data[col])
        data = data.dropna(how='all').dropna(axis=1, how='all')
        data.index = [
            f"{stmt_type}::{str(idx_val).strip()}" for idx_val in data.index
        ]
        self.log.info(
            f"Table {idx}: {stmt_type}, {len(data)} rows, "
            f"cols={list(data.columns)}"
        )
        return data

    def _parse_xlsx(self, file, name: str) -> Optional[pd.DataFrame]:
        file.seek(0)
        xl = pd.ExcelFile(file)
        all_dfs = []
        for sheet in xl.sheet_names:
            try:
                raw = pd.read_excel(file, sheet_name=sheet, header=None)
                if raw.empty or raw.shape[0] < 3:
                    continue
                # Detect units
                if not self.detected_unit:
                    self.detected_unit = UnitDetector.detect_from_headers(raw)
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
                data.index = [
                    f"{stmt_type}::{str(v).strip()}" for v in data.index
                ]
                if not data.empty:
                    all_dfs.append(data)
                    self.log.info(
                        f"Sheet '{sheet}': {stmt_type}, {len(data)} rows"
                    )
            except Exception as e:
                self.log.warning(f"Sheet '{sheet}' error: {e}")
        if all_dfs:
            result = pd.concat(all_dfs, axis=0)
            result = result[~result.index.duplicated(keep='first')]
            return result
        return None

    def _parse_csv(self, file, name: str) -> Optional[pd.DataFrame]:
        file.seek(0)
        for sep in [',', ';', '\t', None]:
            try:
                file.seek(0)
                kw = {'sep': sep, 'header': 0, 'index_col': 0}
                if sep is None:
                    kw['engine'] = 'python'
                df = pd.read_csv(file, **kw)
                if df is not None and not df.empty and len(df.columns) > 0:
                    if not self.detected_unit:
                        self.detected_unit = UnitDetector.detect_from_headers(df)
                    df = self._normalise_columns(df)
                    for col in df.columns:
                        df[col] = self._to_numeric(df[col])
                    df = df.dropna(how='all').dropna(axis=1, how='all')
                    new_idx = []
                    for idx_val in df.index:
                        stype = StatementClassifier.classify(str(idx_val))
                        prefix = (
                            f"{stype}::"
                            if not str(idx_val).startswith(f"{stype}::")
                            else ""
                        )
                        new_idx.append(f"{prefix}{str(idx_val).strip()}")
                    df.index = new_idx
                    self.log.info(f"CSV parsed: {df.shape}")
                    return df
            except Exception:
                continue
        return None

    def _find_header_row(self, df: pd.DataFrame) -> Optional[int]:
        for i in range(min(20, len(df))):
            row = df.iloc[i]
            year_count = sum(
                1 for v in row
                if pd.notna(v) and YearDetector.extract(str(v)) is not None
            )
            if year_count >= 2:
                return i
        return None

    def _find_metric_column(self, df: pd.DataFrame) -> Optional[int]:
        for i in range(min(5, len(df.columns))):
            sample = df.iloc[:min(10, len(df)), i]
            text_count = sum(
                1 for v in sample
                if pd.notna(v) and isinstance(v, str) and len(v.strip()) > 2
            )
            if text_count >= 3:
                return i
        return 0

    def _normalise_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalise column names. FIXED: keep non-year cols if no year cols."""
        new_cols = []
        year_col_indices = []
        for i, col in enumerate(df.columns):
            year = YearDetector.extract(str(col))
            if year:
                new_cols.append(year)
                year_col_indices.append(i)
            else:
                new_cols.append(str(col).strip())
        df.columns = new_cols
        year_cols = [c for c in df.columns if re.match(r'^\d{6}$', str(c))]
        if year_cols:
            # Keep only year columns but preserve index
            non_year = [c for c in df.columns if c not in year_cols]
            # If first col looks like metric names, set as index
            if non_year and len(year_cols) >= 2:
                for nc in non_year:
                    col_data = df[nc]
                    if col_data.dtype == object:
                        text_count = sum(
                            1 for v in col_data[:10]
                            if pd.notna(v) and isinstance(v, str) and len(str(v)) > 2
                        )
                        if text_count >= 3 and nc != df.index.name:
                            df = df.set_index(nc)
                            break
            return df[year_cols]
        return df

    @staticmethod
    def _to_numeric(series: pd.Series) -> pd.Series:
        if series.dtype == 'object' or series.dtype == 'str':
            cleaned = (
                series.astype(str)
                .str.replace(',', '', regex=False)
                .str.replace('₹', '', regex=False)
                .str.replace('$', '', regex=False)
                .str.replace('Rs.', '', regex=False)
                .str.replace('Rs', '', regex=False)
                .str.replace('(', '-', regex=False)
                .str.replace(')', '', regex=False)
                .str.strip()
                .replace({
                    '-': np.nan, '--': np.nan, 'NA': np.nan,
                    'N/A': np.nan, 'nil': '0', 'Nil': '0',
                    'NIL': '0', '': np.nan, 'nan': np.nan,
                    'None': np.nan,
                })
            )
            return pd.to_numeric(cleaned, errors='coerce')
        return pd.to_numeric(series, errors='coerce')


class CompressedFileHandler:
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
                    if p.is_file() and any(
                        p.name.lower().endswith(e) for e in supported
                    ):
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
    valid: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    corrections: List[str] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)

    def add_error(self, msg: str):
        self.errors.append(msg)
        self.valid = False


class DataCleaner:
    def __init__(self):
        self.log = Log.get('Cleaner')

    def clean(
        self, df: pd.DataFrame, winsorize: bool = False
    ) -> Tuple[pd.DataFrame, ValidationReport]:
        report = ValidationReport()
        if df.empty:
            report.add_error("DataFrame is empty")
            return df, report

        out = df.copy()

        # 1. Sanitize index names
        out.index = [
            bleach.clean(str(idx), tags=[], strip=True) for idx in out.index
        ]

        # 2. Deduplicate index
        if out.index.duplicated().any():
            dups = out.index.duplicated().sum()
            report.warnings.append(
                f"{dups} duplicate indices found, keeping first"
            )
            out = out[~out.index.duplicated(keep='first')]

        # 3. Convert all columns to numeric
        for col in out.columns:
            out[col] = CapitalineParser._to_numeric(out[col])

        # 4. Remove fully empty rows/columns
        before = out.shape
        out = out.dropna(how='all').dropna(axis=1, how='all')
        if out.shape != before:
            report.corrections.append(
                f"Removed empty: {before} → {out.shape}"
            )

        # 5. Winsorize outliers (NEW)
        if winsorize:
            num = out.select_dtypes(include=[np.number])
            for col in num.columns:
                s = num[col].dropna()
                if len(s) > 5:
                    lo = s.quantile(Cfg.WINSORIZE_LIMITS[0])
                    hi = s.quantile(Cfg.WINSORIZE_LIMITS[1])
                    clipped = s.clip(lo, hi)
                    changes = (s != clipped).sum()
                    if changes > 0:
                        out[col] = out[col].clip(lo, hi)
                        report.corrections.append(
                            f"Winsorized {changes} values in {col}"
                        )

        # 6. Compute stats
        num = out.select_dtypes(include=[np.number])
        if not num.empty:
            total = num.size
            missing = num.isna().sum().sum()
            report.stats['completeness'] = (
                (1 - missing / total) * 100 if total else 0
            )
            report.stats['shape'] = out.shape
            report.stats['missing_pct'] = (
                missing / total * 100 if total else 0
            )
            report.stats['zero_pct'] = (
                (num == 0).sum().sum() / total * 100 if total else 0
            )
        else:
            report.stats['completeness'] = 0

        return out, report

    def validate_file(self, file: UploadedFile) -> ValidationReport:
        report = ValidationReport()
        if file.size > MAX_FILE_MB * 1024 * 1024:
            report.add_error(
                f"File too large: {file.size / 1e6:.1f}MB > {MAX_FILE_MB}MB"
            )
        ext = Path(file.name).suffix.lower().lstrip('.')
        if ext not in Cfg.ALLOWED_TYPES:
            report.add_error(f"Unsupported type: {ext}")
        suspicious = [r'\.\./', r'[<>"|?*]', r'\.(exe|bat|cmd|sh|ps1)$']
        for pat in suspicious:
            if re.search(pat, file.name, re.I):
                report.add_error("Suspicious filename")
                break
        return report

    def interpolate_missing(
        self, df: pd.DataFrame, method: str = 'linear'
    ) -> pd.DataFrame:
        """NEW: Interpolate missing values in time-series data."""
        num = df.select_dtypes(include=[np.number])
        for col in num.columns:
            if num[col].isna().any() and num[col].notna().sum() >= 2:
                df[col] = df[col].interpolate(method=method, limit=2)
        return df


# ═══════════════════════════════════════════════════════════
# SECTION 8: METRIC PATTERN MATCHING & MAPPING
# ═══════════════════════════════════════════════════════════

class MetricPatterns:
    """
    SIGNIFICANTLY ENHANCED: Added 25+ missing patterns.
    Resolved EBIT/Operating Income collision.
    """

    REGISTRY: Dict[str, Tuple[str, List[Tuple[re.Pattern, float]]]] = {}

    @classmethod
    def _build(cls):
        if cls.REGISTRY:
            return

        defs = {
            # ── Balance Sheet ──
            'Total Assets': ('BalanceSheet', [
                'total assets', 'total equity and liabilities',
                'assets total']),
            'Current Assets': ('BalanceSheet', [
                'current assets', 'total current assets']),
            'Non-Current Assets': ('BalanceSheet', [
                'non-current assets', 'total non-current assets',
                'non current assets', 'fixed assets total']),
            'Cash and Cash Equivalents': ('BalanceSheet', [
                'cash and cash equivalents', 'cash & cash equivalents',
                'cash and bank', 'liquid funds',
                'cash and bank balances', 'balances with banks']),
            'Short-term Investments': ('BalanceSheet', [
                'current investments', 'short term investments',
                'short-term investments', 'other current financial assets',
                'current financial assets']),
            'Long-term Investments': ('BalanceSheet', [
                'non-current investments', 'long term investments',
                'long-term investments', 'other non-current financial assets']),
            'Inventory': ('BalanceSheet', [
                'inventories', 'inventory', 'stock in trade',
                'stock-in-trade', 'finished goods', 'raw materials']),
            'Trade Receivables': ('BalanceSheet', [
                'trade receivables', 'sundry debtors',
                'accounts receivable', 'debtors']),
            'Other Current Assets': ('BalanceSheet', [
                'other current assets', 'loans and advances',
                'short term loans and advances']),
            'Property Plant Equipment': ('BalanceSheet', [
                'property plant and equipment',
                'property, plant and equipment',
                'fixed assets', 'tangible assets', 'net block',
                'gross block']),
            'Goodwill': ('BalanceSheet', [
                'goodwill', 'goodwill on consolidation']),
            'Intangible Assets': ('BalanceSheet', [
                'intangible assets', 'other intangible assets',
                'intangible assets under development']),
            'Right of Use Assets': ('BalanceSheet', [
                'right of use assets', 'right-of-use assets',
                'lease assets']),
            'Capital Work in Progress': ('BalanceSheet', [
                'capital work in progress', 'cwip',
                'capital work-in-progress']),
            'Total Liabilities': ('BalanceSheet', [
                'total liabilities',
                'total non-current liabilities and current liabilities']),
            'Current Liabilities': ('BalanceSheet', [
                'current liabilities', 'total current liabilities']),
            'Non-Current Liabilities': ('BalanceSheet', [
                'non-current liabilities', 'total non-current liabilities',
                'non current liabilities']),
            'Accounts Payable': ('BalanceSheet', [
                'trade payables', 'sundry creditors',
                'accounts payable', 'creditors']),
            'Short-term Debt': ('BalanceSheet', [
                'short term borrowings', 'current borrowings',
                'short-term borrowings',
                'borrowings current', 'current maturities']),
            'Long-term Debt': ('BalanceSheet', [
                'long term borrowings', 'non-current borrowings',
                'long-term borrowings', 'term loans',
                'borrowings non-current']),
            'Other Current Liabilities': ('BalanceSheet', [
                'other current liabilities',
                'other current financial liabilities']),
            'Other Non-Current Liabilities': ('BalanceSheet', [
                'other non-current liabilities',
                'other non-current financial liabilities']),
            'Provisions': ('BalanceSheet', [
                'provisions', 'provision for employee benefits',
                'non-current provisions', 'current provisions']),
            'Deferred Tax Liabilities': ('BalanceSheet', [
                'deferred tax liabilities', 'deferred tax liability',
                'deferred tax liabilities (net)',
                'net deferred tax liabilities']),
            'Deferred Tax Assets': ('BalanceSheet', [
                'deferred tax assets', 'deferred tax asset',
                'net deferred tax assets']),
            'Total Equity': ('BalanceSheet', [
                'total equity', 'shareholders funds',
                "shareholders' funds", 'equity', 'net worth',
                'total shareholders equity']),
            'Share Capital': ('BalanceSheet', [
                'share capital', 'equity share capital',
                'paid-up capital', 'paid up capital']),
            'Retained Earnings': ('BalanceSheet', [
                'reserves and surplus', 'retained earnings',
                'other equity', 'reserves & surplus']),
            'Minority Interest': ('BalanceSheet', [
                'minority interest', 'non-controlling interest',
                'non controlling interests']),
            'Contingent Liabilities': ('BalanceSheet', [
                'contingent liabilities',
                'contingent liabilities and commitments']),

            # ── Profit & Loss ──
            'Revenue': ('ProfitLoss', [
                'revenue from operations',
                'revenue from operations(net)',
                'revenue from operations (net)',
                'total revenue', 'net sales', 'sales', 'revenue',
                'turnover', 'total income from operations']),
            'Cost of Goods Sold': ('ProfitLoss', [
                'cost of materials consumed', 'cost of goods sold',
                'cogs', 'purchase of stock-in-trade',
                'cost of sales', 'material cost',
                'cost of raw materials']),
            'Changes in Inventory': ('ProfitLoss', [
                'changes in inventories',
                'changes in inventories of finished goods',
                'increase/decrease in inventories',
                'change in inventories']),
            'Employee Expenses': ('ProfitLoss', [
                'employee benefit expenses', 'employee benefit expense',
                'employee costs', 'staff costs',
                'salaries and wages', 'personnel expenses']),
            'Operating Expenses': ('ProfitLoss', [
                'other expenses', 'operating expenses',
                'other operating expenses',
                'selling general and administrative']),
            'Total Expenses': ('ProfitLoss', [
                'total expenses', 'total expenditure']),
            'Operating Income': ('ProfitLoss', [
                'profit before exceptional items and tax',
                'operating profit', 'operating income',
                'profit from operations',
                'profit before interest and tax']),
            'EBIT': ('ProfitLoss', [
                'ebit', 'earnings before interest and tax',
                'profit before interest and taxes']),
            'EBITDA': ('ProfitLoss', [
                'ebitda', 'earnings before interest tax depreciation',
                'operating profit before depreciation']),
            'Interest Expense': ('ProfitLoss', [
                'finance cost', 'finance costs', 'interest expense',
                'interest and finance charges', 'borrowing costs',
                'interest paid', 'financial expenses']),
            'Interest Income': ('ProfitLoss', [
                'interest income', 'interest received',
                'income from investments']),
            'Other Income': ('ProfitLoss', [
                'other income', 'other operating income',
                'miscellaneous income', 'non-operating income']),
            'Exceptional Items': ('ProfitLoss', [
                'exceptional items', 'exceptional item',
                'extraordinary items', 'prior period items']),
            'Income Before Tax': ('ProfitLoss', [
                'profit before tax', 'pbt', 'income before tax',
                'profit/(loss) before tax']),
            'Tax Expense': ('ProfitLoss', [
                'tax expense', 'tax expenses', 'current tax',
                'total tax expense', 'income tax',
                'provision for tax', 'tax provision']),
            'Net Income': ('ProfitLoss', [
                'profit after tax',
                'profit/loss for the period',
                'profit/(loss) for the period',
                'net profit', 'pat', 'net income',
                'profit for the period',
                'profit for the year']),
            'Depreciation': ('ProfitLoss', [
                'depreciation and amortisation expenses',
                'depreciation and amortization',
                'depreciation', 'depreciation & amortisation',
                'depreciation and amortisation']),
            'EPS Basic': ('ProfitLoss', [
                'basic eps', 'earnings per share basic',
                'basic earnings per share', 'eps basic']),
            'EPS Diluted': ('ProfitLoss', [
                'diluted eps', 'earnings per share diluted',
                'diluted earnings per share', 'eps diluted']),
            'Dividend Per Share': ('ProfitLoss', [
                'dividend per share', 'dividend per equity share',
                'dps', 'proposed dividend']),
            'Total Comprehensive Income': ('ProfitLoss', [
                'total comprehensive income',
                'other comprehensive income',
                'total comprehensive income for the period']),

            # ── Cash Flow ──
            'Operating Cash Flow': ('CashFlow', [
                'net cash from operating activities',
                'net cashflow from operating activities',
                'operating cash flow',
                'cash from operating activities',
                'cash generated from operations',
                'net cash generated from operating activities']),
            'Capital Expenditure': ('CashFlow', [
                'purchase of fixed assets',
                'purchased of fixed assets',
                'capital expenditure',
                'additions to fixed assets',
                'purchase of property plant and equipment',
                'acquisition of fixed assets']),
            'Purchase of Investments': ('CashFlow', [
                'purchase of investments',
                'purchase of non-current investments',
                'investment in subsidiaries']),
            'Sale of Investments': ('CashFlow', [
                'sale of investments', 'proceeds from sale of investments',
                'sale of non-current investments']),
            'Investing Cash Flow': ('CashFlow', [
                'net cash used in investing',
                'cash flow from investing',
                'net cash from investing activities',
                'net cashflow from investing activities']),
            'Financing Cash Flow': ('CashFlow', [
                'net cash used in financing',
                'cash flow from financing',
                'net cash from financing activities',
                'net cashflow from financing activities']),
            'Dividends Paid': ('CashFlow', [
                'dividends paid', 'dividend paid',
                'payment of dividends']),
            'Debt Repayment': ('CashFlow', [
                'repayment of borrowings',
                'repayment of long term borrowings',
                'repayment of loans']),
            'Debt Proceeds': ('CashFlow', [
                'proceeds from borrowings',
                'proceeds from long term borrowings',
                'proceeds from loans']),
            'Net Change in Cash': ('CashFlow', [
                'net increase in cash',
                'net increase/(decrease) in cash',
                'net change in cash',
                'increase in cash and cash equivalents']),
        }

        for target, (stmt, patterns) in defs.items():
            compiled = [
                (re.compile(re.escape(p), re.I), 1.0) for p in patterns
            ]
            cls.REGISTRY[target] = (stmt, compiled)

    @classmethod
    def match(cls, metric_name: str) -> List[Tuple[str, float]]:
        cls._build()
        clean = (
            metric_name.split('::')[-1].strip().lower()
            if '::' in metric_name
            else metric_name.strip().lower()
        )
        results = []
        for target, (stmt, patterns) in cls.REGISTRY.items():
            best_score = 0
            for pat, weight in patterns:
                if pat.search(clean):
                    pat_text = pat.pattern.replace('\\', '').lower()
                    score = 0.95 if pat_text == clean else 0.80
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

    @classmethod
    def get_all_targets(cls) -> List[str]:
        cls._build()
        return list(cls.REGISTRY.keys())

    @classmethod
    def get_targets_by_statement(cls) -> Dict[str, List[str]]:
        cls._build()
        result = defaultdict(list)
        for target, (stmt, _) in cls.REGISTRY.items():
            result[stmt].append(target)
        return dict(result)


class MappingTemplate:
    @staticmethod
    def create_auto_mapping(
        source_metrics: List[str]
    ) -> Tuple[Dict[str, str], List[str]]:
        MetricPatterns._build()
        mappings = {}
        unmapped = []
        used_targets: Set[str] = set()

        # Pass 1: Pattern matching with statement validation
        scored = []
        for source in source_metrics:
            matches = MetricPatterns.match(source)
            source_stmt = (
                source.split('::')[0] if '::' in source else None
            )
            for target, conf in matches:
                required_stmt = MetricPatterns.get_required_statement(target)
                if (source_stmt and required_stmt and
                        source_stmt != required_stmt):
                    continue
                scored.append((source, target, conf))

        # Sort by confidence desc
        scored.sort(key=lambda x: x[2], reverse=True)
        used_sources: Set[str] = set()
        for source, target, conf in scored:
            if source in used_sources or target in used_targets:
                continue
            if conf >= 0.6:
                mappings[source] = target
                used_sources.add(source)
                used_targets.add(target)

        # Pass 2: Fuzzy matching for remaining
        remaining = [
            s for s in source_metrics if s not in mappings
        ]
        all_targets = list(MetricPatterns.REGISTRY.keys())
        available_targets = [t for t in all_targets if t not in used_targets]

        for source in remaining:
            clean = (
                source.split('::')[-1].strip()
                if '::' in source
                else source.strip()
            )
            best_target, best_score = None, 0
            for target in available_targets:
                score = fuzz.token_sort_ratio(
                    clean.lower(), target.lower()
                ) / 100
                if score > best_score:
                    best_score = score
                    best_target = target
            if best_target and best_score >= 0.70:
                mappings[source] = best_target
                available_targets.remove(best_target)
            else:
                unmapped.append(source)

        return mappings, unmapped


# ═══════════════════════════════════════════════════════════
# SECTION 9: KAGGLE API CLIENT
# ═══════════════════════════════════════════════════════════

class KaggleClient:
    def __init__(
        self, base_url: str, timeout: int = 30, retries: int = 3
    ):
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
            max_retries=Retry(
                total=retries, backoff_factor=1,
                status_forcelist=[429, 500, 502, 503, 504]
            ),
            pool_connections=10, pool_maxsize=20
        )
        self._session.mount("http://", adapter)
        self._session.mount("https://", adapter)
        self._session.headers.update({
            'Content-Type': 'application/json',
            'ngrok-skip-browser-warning': 'true'
        })

    def health_check(self) -> bool:
        if not self._session:
            return False
        try:
            r = self._session.post(
                f"{self.base_url}/embed",
                json={'texts': ['test']},
                timeout=10, verify=False
            )
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
            results = []
            for i in range(0, len(texts), Cfg.KAGGLE_BATCH):
                batch = texts[i:i + Cfg.KAGGLE_BATCH]
                r = self._session.post(
                    f"{self.base_url}/embed",
                    json={'texts': batch},
                    timeout=self.timeout, verify=False
                )
                if r.status_code == 200:
                    data = r.json()
                    if 'embeddings' in data:
                        results.extend(
                            [np.array(e) for e in data['embeddings']]
                        )
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
    def __init__(self):
        self.log = Log.get('Mapper')
        self._model = None
        self._kaggle: Optional[KaggleClient] = None
        self._embed_cache = Cache(max_entries=2000, ttl=7200)
        self._std_embeddings: Dict[str, np.ndarray] = {}

    def initialize(self, kaggle_url: str = ''):
        if kaggle_url and REQUESTS_OK:
            self._kaggle = KaggleClient(kaggle_url)
            if self._kaggle.health_check():
                self.log.info("Kaggle GPU connected")
            else:
                self.log.warning("Kaggle unavailable, falling back")
        if ST_OK and (
            not self._kaggle or not self._kaggle.is_available
        ):
            try:
                self._model = SentenceTransformer(Cfg.AI_MODEL)
                self.log.info(f"Local model loaded: {Cfg.AI_MODEL}")
            except Exception as e:
                self.log.warning(f"Local model failed: {e}")
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
                emb = self._model.encode(
                    text, convert_to_numpy=True,
                    show_progress_bar=False
                )
            except Exception:
                pass
        if emb is not None:
            self._embed_cache.put(key, emb)
        return emb

    def map_metrics(
        self, sources: List[str], threshold: float = 0.6
    ) -> Dict[str, Any]:
        if not self._model and not (
            self._kaggle and self._kaggle.is_available
        ):
            mappings, unmapped = MappingTemplate.create_auto_mapping(sources)
            return {
                'mappings': mappings,
                'confidence': {s: 0.8 for s in mappings},
                'unmapped': unmapped,
                'method': 'pattern+fuzzy'
            }

        mappings = {}
        confidence = {}
        unmapped = []
        used_targets: Set[str] = set()

        for source in sources:
            clean = (
                source.split('::')[-1].strip()
                if '::' in source else source.strip()
            )
            src_emb = self._get_embedding(clean.lower())

            if src_emb is None:
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

            best_target, best_score = None, 0
            for target, tgt_emb in self._std_embeddings.items():
                if target in used_targets:
                    continue
                sim = float(cosine_similarity(
                    src_emb.reshape(1, -1),
                    tgt_emb.reshape(1, -1)
                )[0, 0])
                if sim > best_score:
                    best_score = sim
                    best_target = target

            if best_target and best_score >= threshold:
                mappings[source] = best_target
                confidence[source] = best_score
                used_targets.add(best_target)
            else:
                unmapped.append(source)

        method = (
            'kaggle_ai'
            if (self._kaggle and self._kaggle.is_available)
            else 'local_ai'
        )
        return {
            'mappings': mappings, 'confidence': confidence,
            'unmapped': unmapped, 'method': method
        }

    def get_status(self) -> Dict:
        return {
            'kaggle_ok': (
                self._kaggle.is_available if self._kaggle else False
            ),
            'local_ok': self._model is not None,
            'cache_entries': len(self._embed_cache._store),
            'cache_hit_rate': self._embed_cache.hit_rate,
        }


# ═══════════════════════════════════════════════════════════
# SECTION 11: FINANCIAL ANALYSIS ENGINE (ENHANCED)
# ═══════════════════════════════════════════════════════════

class FinancialAnalyzer:
    """Enhanced with DuPont, Working Capital, Correlation."""

    def __init__(self):
        self.log = Log.get('Analyzer')
        self._cache = Cache(max_entries=20, ttl=3600)

    def analyze(self, df: pd.DataFrame) -> Dict[str, Any]:
        key = hashlib.md5(
            str(df.shape).encode() +
            str(df.index[:5].tolist()).encode()
        ).hexdigest()
        cached = self._cache.get(key)
        if cached:
            return cached

        with Perf.measure('full_analysis'):
            result = {
                'summary': self._summary(df),
                'ratios': self._ratios(df),
                'trends': self._trends(df),
                'anomalies': self._anomalies(df),
                'working_capital': self._working_capital(df),
                'dupont': self._dupont_analysis(df),
                'correlation': self._correlation(df),
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
            'year_range': (
                f"{num.columns[0]}–{num.columns[-1]}"
                if not num.empty and len(num.columns) > 0 else 'N/A'
            ),
            'completeness': (1 - missing / total_cells) * 100,
            'statement_breakdown': self._statement_breakdown(df),
        }

    def _statement_breakdown(self, df: pd.DataFrame) -> Dict[str, int]:
        """NEW: Count metrics by statement type."""
        breakdown = defaultdict(int)
        for idx in df.index:
            parts = str(idx).split('::')
            stmt = parts[0] if len(parts) > 1 else 'Other'
            breakdown[stmt] += 1
        return dict(breakdown)

    def _ratios(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        ratios = {}

        def _find(keyword: str) -> Optional[pd.Series]:
            for idx in df.index:
                if keyword.lower() in str(idx).lower():
                    s = df.loc[idx]
                    return s.iloc[0] if isinstance(s, pd.DataFrame) else s
            return None

        def _safe_div(
            num_s: Optional[pd.Series], den_s: Optional[pd.Series]
        ) -> Optional[pd.Series]:
            if num_s is None or den_s is None:
                return None
            return num_s / den_s.replace(0, np.nan)

        ca = _find('current assets')
        cl = _find('current liabilities')
        ta = _find('total assets')
        tl = _find('total liabilities')
        te = _find('total equity')
        rev = _find('revenue')
        ni = _find('net income') or _find('profit after tax')
        inv = _find('inventor')
        ebit = _find('ebit') or _find('operating profit')
        ie = _find('interest expense') or _find('finance cost')
        cogs = _find('cost of goods') or _find('cost of materials')
        recv = _find('receivable')
        dep = _find('depreciation')

        # Liquidity
        liq = {}
        cr = _safe_div(ca, cl)
        if cr is not None:
            liq['Current Ratio'] = cr
        if ca is not None and inv is not None and cl is not None:
            liq['Quick Ratio'] = (ca - inv) / cl.replace(0, np.nan)
        if ca is not None and ta is not None:
            cash = _find('cash')
            if cash is not None and cl is not None:
                liq['Cash Ratio'] = cash / cl.replace(0, np.nan)
        ratios['Liquidity'] = pd.DataFrame(liq).T if liq else pd.DataFrame()

        # Profitability
        prof = {}
        npm = _safe_div(ni, rev)
        if npm is not None:
            prof['Net Profit Margin %'] = npm * 100
        gp = None
        if rev is not None and cogs is not None:
            gp = rev - cogs
            prof['Gross Profit Margin %'] = (gp / rev.replace(0, np.nan)) * 100
        roa = _safe_div(ni, ta)
        if roa is not None:
            prof['ROA %'] = roa * 100
        roe = _safe_div(ni, te)
        if roe is not None:
            prof['ROE %'] = roe * 100
        if ebit is not None and rev is not None:
            prof['Operating Margin %'] = (
                ebit / rev.replace(0, np.nan)
            ) * 100
        if ebit is not None and dep is not None and rev is not None:
            ebitda = ebit + dep
            prof['EBITDA Margin %'] = (
                ebitda / rev.replace(0, np.nan)
            ) * 100
        ratios['Profitability'] = (
            pd.DataFrame(prof).T if prof else pd.DataFrame()
        )

        # Leverage
        lev = {}
        de = _safe_div(tl, te)
        if de is not None:
            lev['Debt/Equity'] = de
        icr = _safe_div(ebit, ie)
        if icr is not None:
            lev['Interest Coverage'] = icr
        if ta is not None and te is not None:
            lev['Equity Multiplier'] = ta / te.replace(0, np.nan)
        ratios['Leverage'] = pd.DataFrame(lev).T if lev else pd.DataFrame()

        # Efficiency (NEW)
        eff = {}
        if rev is not None and ta is not None:
            eff['Asset Turnover'] = rev / ta.replace(0, np.nan)
        if cogs is not None and inv is not None:
            eff['Inventory Turnover'] = cogs / inv.replace(0, np.nan)
            days = inv / (cogs / 365).replace(0, np.nan)
            eff['Days Inventory Outstanding'] = days
        if rev is not None and recv is not None:
            eff['Receivable Turnover'] = rev / recv.replace(0, np.nan)
            eff['Days Sales Outstanding'] = (
                recv / (rev / 365).replace(0, np.nan)
            )
        ap = _find('payable')
        if cogs is not None and ap is not None:
            eff['Days Payable Outstanding'] = (
                ap / (cogs / 365).replace(0, np.nan)
            )
        # Cash Conversion Cycle
        if all(k in eff for k in [
            'Days Inventory Outstanding',
            'Days Sales Outstanding',
            'Days Payable Outstanding'
        ]):
            eff['Cash Conversion Cycle'] = (
                eff['Days Inventory Outstanding'] +
                eff['Days Sales Outstanding'] -
                eff['Days Payable Outstanding']
            )
        ratios['Efficiency'] = (
            pd.DataFrame(eff).T if eff else pd.DataFrame()
        )

        return {k: v for k, v in ratios.items() if not v.empty}

    def _working_capital(self, df: pd.DataFrame) -> Dict:
        """NEW: Working capital analysis."""
        def _find(kw):
            for idx in df.index:
                if kw.lower() in str(idx).lower():
                    s = df.loc[idx]
                    return s.iloc[0] if isinstance(s, pd.DataFrame) else s
            return None

        result = {}
        ca = _find('current assets')
        cl = _find('current liabilities')
        if ca is not None and cl is not None:
            wc = ca - cl
            result['Working Capital'] = wc
            result['WC Change'] = wc.diff()
            rev = _find('revenue')
            if rev is not None:
                result['WC/Revenue %'] = (
                    wc / rev.replace(0, np.nan) * 100
                )
        return result

    def _dupont_analysis(self, df: pd.DataFrame) -> Dict:
        """NEW: DuPont 3-factor and 5-factor decomposition."""
        def _find(kw):
            for idx in df.index:
                if kw.lower() in str(idx).lower():
                    s = df.loc[idx]
                    return s.iloc[0] if isinstance(s, pd.DataFrame) else s
            return None

        result = {}
        ni = _find('net income') or _find('profit after tax')
        rev = _find('revenue')
        ta = _find('total assets')
        te = _find('total equity')

        if all(v is not None for v in [ni, rev, ta, te]):
            npm = ni / rev.replace(0, np.nan)
            at = rev / ta.replace(0, np.nan)
            em = ta / te.replace(0, np.nan)
            result['3_factor'] = {
                'Net Profit Margin': npm,
                'Asset Turnover': at,
                'Equity Multiplier': em,
                'ROE (computed)': npm * at * em * 100,
            }

            # 5-factor
            pbt = _find('profit before tax')
            ebit = _find('ebit') or _find('operating profit')
            if pbt is not None and ebit is not None:
                tax_burden = ni / pbt.replace(0, np.nan)
                interest_burden = pbt / ebit.replace(0, np.nan)
                opm = ebit / rev.replace(0, np.nan)
                result['5_factor'] = {
                    'Tax Burden': tax_burden,
                    'Interest Burden': interest_burden,
                    'Operating Margin': opm,
                    'Asset Turnover': at,
                    'Equity Multiplier': em,
                    'ROE (computed)': (
                        tax_burden * interest_burden * opm * at * em * 100
                    ),
                }

        return result

    def _correlation(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """NEW: Cross-metric correlation matrix."""
        num = df.select_dtypes(include=[np.number])
        if num.shape[0] < 3 or num.shape[1] < 3:
            return None
        # Transpose: metrics as columns for correlation
        t = num.T
        # Only keep metrics with enough data
        valid = t.columns[t.notna().sum() >= 3]
        if len(valid) < 3:
            return None
        corr = t[valid].corr()
        # Limit to top 20 metrics for readability
        if len(corr) > 20:
            # Select metrics with highest variance
            variances = t[valid].var().nlargest(20)
            corr = corr.loc[variances.index, variances.index]
        return corr

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
            cagr = 0
            if s.iloc[0] > 0 and s.iloc[-1] > 0 and len(s) > 1:
                cagr = (
                    (s.iloc[-1] / s.iloc[0]) ** (1 / (len(s) - 1)) - 1
                ) * 100
            yoy_changes = s.pct_change().dropna() * 100
            trends[str(idx)] = {
                'direction': 'increasing' if slope > 0 else 'decreasing',
                'cagr': round(cagr, 2),
                'volatility': round(float(s.pct_change().std() * 100), 2),
                'yoy_growth': yoy_changes.to_dict(),
                'latest_value': float(s.iloc[-1]),
                'min_value': float(s.min()),
                'max_value': float(s.max()),
            }
        return trends

    def _anomalies(self, df: pd.DataFrame) -> Dict:
        anomalies = {'value': [], 'trend': [], 'sign_change': []}
        num = df.select_dtypes(include=[np.number])
        for idx in num.index:
            s = num.loc[idx]
            if isinstance(s, pd.DataFrame):
                s = s.iloc[0]
            s = s.dropna()
            if len(s) < 4:
                continue
            # IQR outliers
            Q1, Q3 = s.quantile(0.25), s.quantile(0.75)
            IQR = Q3 - Q1
            if IQR > 0:
                outliers = s[(s < Q1 - 3 * IQR) | (s > Q3 + 3 * IQR)]
                for year, val in outliers.items():
                    anomalies['value'].append({
                        'metric': str(idx), 'year': str(year),
                        'value': float(val)
                    })
            # Sign changes (NEW)
            signs = np.sign(s.values)
            for i in range(1, len(signs)):
                if signs[i] != signs[i - 1] and signs[i] != 0:
                    anomalies['sign_change'].append({
                        'metric': str(idx),
                        'year': str(s.index[i]),
                        'from': float(s.iloc[i - 1]),
                        'to': float(s.iloc[i]),
                    })
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
            insights.append(
                "⚠️ Low data quality — consider checking mappings"
            )

        breakdown = summary.get('statement_breakdown', {})
        if breakdown:
            types = list(breakdown.keys())
            if 'CashFlow' not in types:
                insights.append(
                    "⚠️ Cash flow statement missing — "
                    "FCF analysis unavailable"
                )
            if 'BalanceSheet' not in types:
                insights.append(
                    "⚠️ Balance sheet missing — "
                    "leverage ratios unavailable"
                )

        trends = analysis.get('trends', {})
        rev_trends = [v for k, v in trends.items() if 'revenue' in k.lower()]
        if rev_trends:
            cagr = rev_trends[0].get('cagr', 0)
            if cagr > 15:
                insights.append(
                    f"🚀 Strong revenue growth (CAGR: {cagr:.1f}%)"
                )
            elif cagr < 0:
                insights.append(
                    f"📉 Declining revenue (CAGR: {cagr:.1f}%)"
                )

        # Working capital insight (NEW)
        wc = analysis.get('working_capital', {})
        if 'WC/Revenue %' in wc:
            wc_ratio = wc['WC/Revenue %']
            if hasattr(wc_ratio, 'iloc') and len(wc_ratio) > 0:
                latest = wc_ratio.iloc[-1]
                if pd.notna(latest):
                    if latest < 0:
                        insights.append(
                            "❌ Negative working capital — "
                            "liquidity risk"
                        )
                    elif latest > 50:
                        insights.append(
                            "⚠️ High working capital ratio — "
                            "capital may be underutilised"
                        )

        # Anomaly insight
        anomalies = analysis.get('anomalies', {})
        sign_changes = anomalies.get('sign_change', [])
        if sign_changes:
            insights.append(
                f"🔄 {len(sign_changes)} sign change(s) detected "
                f"— review for unusual items"
            )

        return insights


# ═══════════════════════════════════════════════════════════
# SECTION 12: PENMAN-NISSIM ANALYZER (ENHANCED)
# ═══════════════════════════════════════════════════════════

class PenmanNissimAnalyzer:
    """
    Enhanced Penman-Nissim with:
    - Residual Income / EVA
    - Sustainable Growth Rate
    - DuPont integration
    - Better debt-free handling
    """

    def __init__(self, df: pd.DataFrame, mappings: Dict[str, str]):
        self.log = Log.get('PenmanNissim')
        self._raw = df
        self._mappings = mappings
        self._inv_map = {v: k for k, v in mappings.items()}
        self._data = self._restructure(df)
        self._ref_bs: Optional[pd.DataFrame] = None
        self._ref_is: Optional[pd.DataFrame] = None

    def _get(self, target: str, default_zero: bool = False) -> pd.Series:
        source = self._inv_map.get(target)
        if source and source in self._data.index:
            s = self._data.loc[source]
            if isinstance(s, pd.DataFrame):
                s = s.iloc[0]
            return s.fillna(0) if default_zero else s

        # Derived metrics
        if target == 'Total Liabilities':
            try:
                return self._get('Total Assets') - self._get('Total Equity')
            except Exception:
                pass
        if target == 'EBIT':
            try:
                pbt = self._get('Income Before Tax')
                fc = self._get('Interest Expense', True)
                return pbt + fc
            except Exception:
                pass
        if target == 'EBITDA':
            try:
                ebit = self._get('EBIT')
                dep = self._get('Depreciation', True)
                return ebit + dep
            except Exception:
                pass
        if target == 'Total Expenses':
            try:
                rev = self._get('Revenue')
                pbt = self._get('Income Before Tax')
                oi = self._get('Other Income', True)
                return rev + oi - pbt
            except Exception:
                pass

        if default_zero:
            return pd.Series(0, index=self._data.columns, dtype=float)
        raise ValueError(f"Metric '{target}' not found in mappings")

    def _has(self, target: str) -> bool:
        if target in self._inv_map:
            return True
        # Check derived
        derivable = {
            'Total Liabilities': {'Total Assets', 'Total Equity'},
            'EBIT': {'Income Before Tax', 'Interest Expense'},
            'EBITDA': {'Income Before Tax', 'Interest Expense', 'Depreciation'},
        }
        if target in derivable:
            return all(self._has(dep) for dep in derivable[target])
        return False

    def _restructure(self, df: pd.DataFrame) -> pd.DataFrame:
        year_map = YearDetector.detect_columns(df)
        if not year_map:
            self.log.warning("No year columns detected, using raw")
            return df.apply(pd.to_numeric, errors='coerce')

        years = sorted(year_map.keys())
        out = pd.DataFrame(index=df.index, columns=years, dtype=float)

        for year, src_cols in year_map.items():
            for idx in df.index:
                for col in src_cols:
                    try:
                        val = df.loc[idx, col]
                        if pd.notna(val):
                            num = float(
                                str(val).replace(',', '')
                                .replace('(', '-')
                                .replace(')', '').strip()
                            )
                            if (pd.isna(out.loc[idx, year]) or
                                    out.loc[idx, year] == 0):
                                out.loc[idx, year] = num
                    except (ValueError, TypeError):
                        continue

        self.log.info(f"Restructured: {out.shape}, years={years}")
        return out

    def reformulate_balance_sheet(self) -> pd.DataFrame:
        if self._ref_bs is not None:
            return self._ref_bs

        self.log.info("=== Reformulating Balance Sheet ===")
        cols = self._data.columns
        ref = pd.DataFrame(index=cols, dtype=float)

        total_assets = self._get('Total Assets')
        total_equity = self._get('Total Equity')
        total_liab = self._get('Total Liabilities', default_zero=True)

        # Financial Assets
        cash = self._get('Cash and Cash Equivalents', True)
        curr_inv = (
            self._get('Short-term Investments', True)
            if self._has('Short-term Investments')
            else pd.Series(0, index=cols)
        )
        lt_inv = (
            self._get('Long-term Investments', True)
            if self._has('Long-term Investments')
            else pd.Series(0, index=cols)
        )
        financial_assets = cash + curr_inv + lt_inv

        # Financial Liabilities
        st_debt = self._get('Short-term Debt', True)
        lt_debt = self._get('Long-term Debt', True)
        financial_liab = st_debt + lt_debt

        nfa = financial_assets - financial_liab

        operating_assets = total_assets - financial_assets
        operating_liab = (total_liab - financial_liab).clip(lower=0)
        noa = operating_assets - operating_liab

        # Minority interest handling (NEW)
        minority = (
            self._get('Minority Interest', True)
            if self._has('Minority Interest')
            else pd.Series(0, index=cols)
        )

        ref['Total Assets'] = total_assets
        ref['Operating Assets'] = operating_assets
        ref['Financial Assets'] = financial_assets
        ref['Total Liabilities'] = total_liab
        ref['Operating Liabilities'] = operating_liab
        ref['Financial Liabilities'] = financial_liab
        ref['Net Operating Assets'] = noa
        ref['Net Financial Assets'] = nfa
        ref['Net Financial Obligations'] = -nfa
        ref['Common Equity'] = total_equity
        ref['Minority Interest'] = minority
        ref['Total Debt'] = financial_liab
        ref['Net Debt'] = financial_liab - cash
        ref['Cash and Equivalents'] = cash
        ref['Invested Capital'] = noa + cash

        # NEW: Working Capital components
        ca = self._get('Current Assets', True)
        cl = self._get('Current Liabilities', True)
        ref['Current Assets'] = ca
        ref['Current Liabilities'] = cl
        ref['Net Working Capital'] = ca - cl

        # Validate
        check = (noa + nfa - total_equity).abs()
        if check.max() > 1:
            self.log.warning(
                f"BS check: max imbalance = {check.max():.0f}"
            )

        self._ref_bs = ref.T
        return self._ref_bs

    def reformulate_income_statement(self) -> pd.DataFrame:
        if self._ref_is is not None:
            return self._ref_is

        self.log.info("=== Reformulating Income Statement ===")
        cols = self._data.columns
        ref = pd.DataFrame(index=cols, dtype=float)

        revenue = self._get('Revenue')
        pbt = self._get('Income Before Tax', True)
        tax = self._get('Tax Expense', True)
        finance_cost = self._get('Interest Expense', True)
        other_income = (
            self._get('Other Income', True)
            if self._has('Other Income')
            else pd.Series(0, index=cols)
        )
        net_income = self._get('Net Income')

        ebit = pbt + finance_cost
        operating_income_bt = ebit - other_income

        # Effective tax rate
        eff_tax = pd.Series(0.25, index=cols)
        for year in cols:
            if (pd.notna(pbt[year]) and pbt[year] > 0 and
                    pd.notna(tax[year])):
                rate = tax[year] / pbt[year]
                if 0.05 <= rate <= 0.5:
                    eff_tax[year] = rate

        # Tax allocation
        tax_on_operating = operating_income_bt * eff_tax
        tax_on_financial = (other_income - finance_cost) * eff_tax
        tax_allocated = tax_on_operating + tax_on_financial
        for year in cols:
            if pd.notna(tax[year]) and abs(tax_allocated[year]) > EPS:
                adj = tax[year] - tax_allocated[year]
                tax_on_operating[year] += adj

        oi_at = operating_income_bt - tax_on_operating
        nfi_at = (other_income - finance_cost) - tax_on_financial

        ref['Revenue'] = revenue
        ref['EBIT'] = ebit
        ref['Operating Income Before Tax'] = operating_income_bt
        ref['Tax on Operating Income'] = tax_on_operating
        ref['Operating Income After Tax (NOPAT)'] = oi_at
        ref['NOPAT'] = oi_at
        ref['Net Financial Income Before Tax'] = (
            other_income - finance_cost
        )
        ref['Net Financial Income After Tax'] = nfi_at
        ref['Net Financial Expense After Tax'] = -nfi_at
        ref['Interest Expense'] = finance_cost
        ref['Other Income'] = other_income
        ref['Effective Tax Rate'] = eff_tax
        ref['Net Income (Reported)'] = net_income
        ref['Net Income (Calculated)'] = oi_at + nfi_at
        ref['Tax Expense'] = tax

        if self._has('Cost of Goods Sold'):
            cogs = self._get('Cost of Goods Sold', True)
            ref['Cost of Goods Sold'] = cogs
            ref['Gross Profit'] = revenue - cogs
            ref['Gross Profit Margin %'] = (
                (revenue - cogs) / revenue.replace(0, np.nan) * 100
            )

        if self._has('Depreciation'):
            dep = self._get('Depreciation', True)
            ref['EBITDA'] = operating_income_bt + dep
            ref['Depreciation'] = dep

        if self._has('Employee Expenses'):
            emp = self._get('Employee Expenses', True)
            ref['Employee Expenses'] = emp

        if self._has('Operating Expenses'):
            opex = self._get('Operating Expenses', True)
            ref['Other Operating Expenses'] = opex

        if self._has('Exceptional Items'):
            exc = self._get('Exceptional Items', True)
            ref['Exceptional Items'] = exc

        if self._has('EPS Basic'):
            eps = self._get('EPS Basic', True)
            ref['EPS Basic'] = eps

        if self._has('Dividend Per Share'):
            dps = self._get('Dividend Per Share', True)
            ref['Dividend Per Share'] = dps

        self.log.info("IS reformulation complete")
        self._ref_is = ref.T
        return self._ref_is

    def calculate_ratios(self) -> pd.DataFrame:
        bs = self.reformulate_balance_sheet()
        is_ = self.reformulate_income_statement()
        years = bs.columns

        ratios = pd.DataFrame(index=years, dtype=float)

        def _avg(series_name: str, source: pd.DataFrame) -> pd.Series:
            s = (source.loc[series_name]
                 if series_name in source.index
                 else pd.Series(0, index=years))
            avg = pd.Series(index=years, dtype=float)
            for i, y in enumerate(years):
                avg[y] = (
                    s[y] if i == 0
                    else (s[years[i - 1]] + s[y]) / 2
                )
            return avg

        # RNOA
        nopat = (
            is_.loc['NOPAT']
            if 'NOPAT' in is_.index
            else pd.Series(np.nan, index=years)
        )
        avg_noa = _avg('Net Operating Assets', bs)

        rnoa = pd.Series(np.nan, index=years)
        for y in years:
            if (pd.notna(avg_noa[y]) and abs(avg_noa[y]) > 10 and
                    pd.notna(nopat[y])):
                rnoa[y] = (nopat[y] / avg_noa[y]) * 100
        ratios['RNOA %'] = rnoa

        # OPM & NOAT
        revenue = (
            is_.loc['Revenue']
            if 'Revenue' in is_.index
            else pd.Series(np.nan, index=years)
        )
        opm = pd.Series(np.nan, index=years)
        noat = pd.Series(np.nan, index=years)
        for y in years:
            if pd.notna(revenue[y]) and revenue[y] > 0 and pd.notna(nopat[y]):
                opm[y] = (nopat[y] / revenue[y]) * 100
            if (pd.notna(avg_noa[y]) and abs(avg_noa[y]) > 10 and
                    pd.notna(revenue[y])):
                noat[y] = revenue[y] / avg_noa[y]
        ratios['Operating Profit Margin (OPM) %'] = opm
        ratios['Net Operating Asset Turnover (NOAT)'] = noat

        # FLEV
        nfa = (
            bs.loc['Net Financial Assets']
            if 'Net Financial Assets' in bs.index
            else pd.Series(0, index=years)
        )
        avg_ce = _avg('Common Equity', bs)

        flev = pd.Series(np.nan, index=years)
        for y in years:
            if pd.notna(avg_ce[y]) and abs(avg_ce[y]) > 10 and pd.notna(nfa[y]):
                flev[y] = -nfa[y] / avg_ce[y]
        ratios['Financial Leverage (FLEV)'] = flev

        # NBC
        nfe_at = (
            is_.loc['Net Financial Expense After Tax']
            if 'Net Financial Expense After Tax' in is_.index
            else pd.Series(0, index=years)
        )
        avg_nfo = -_avg('Net Financial Assets', bs)
        total_debt = (
            bs.loc['Total Debt']
            if 'Total Debt' in bs.index
            else pd.Series(0, index=years)
        )

        nbc = pd.Series(0.0, index=years)
        for y in years:
            if (pd.notna(avg_nfo[y]) and abs(avg_nfo[y]) > 10 and
                    pd.notna(nfe_at[y])):
                nbc[y] = np.clip(
                    (nfe_at[y] / avg_nfo[y]) * 100, -15, 25
                )
            elif total_debt[y] <= 10:
                nbc[y] = 0
        ratios['Net Borrowing Cost (NBC) %'] = nbc

        # Spread
        spread = rnoa - nbc
        ratios['Spread %'] = spread

        # ROE
        net_income = (
            is_.loc['Net Income (Reported)']
            if 'Net Income (Reported)' in is_.index
            else pd.Series(np.nan, index=years)
        )
        roe = pd.Series(np.nan, index=years)
        for y in years:
            if (pd.notna(avg_ce[y]) and abs(avg_ce[y]) > 10 and
                    pd.notna(net_income[y])):
                roe[y] = (net_income[y] / avg_ce[y]) * 100
        ratios['ROE %'] = roe
        ratios['ROE (PN Decomposed) %'] = rnoa + flev * spread

        # ROA
        avg_ta = _avg('Total Assets', bs)
        roa = pd.Series(np.nan, index=years)
        for y in years:
            if (pd.notna(avg_ta[y]) and avg_ta[y] > 0 and
                    pd.notna(net_income[y])):
                roa[y] = (net_income[y] / avg_ta[y]) * 100
        ratios['ROA %'] = roa

        # ROIC (NEW)
        avg_ic = _avg('Invested Capital', bs)
        roic = pd.Series(np.nan, index=years)
        for y in years:
            if (pd.notna(avg_ic[y]) and avg_ic[y] > 10 and
                    pd.notna(nopat[y])):
                roic[y] = (nopat[y] / avg_ic[y]) * 100
        ratios['ROIC %'] = roic

        # Margins
        if 'Gross Profit' in is_.index:
            gp = is_.loc['Gross Profit']
            gpm = pd.Series(np.nan, index=years)
            for y in years:
                if pd.notna(revenue[y]) and revenue[y] > 0:
                    gpm[y] = (gp[y] / revenue[y]) * 100
            ratios['Gross Profit Margin %'] = gpm

        npm = pd.Series(np.nan, index=years)
        for y in years:
            if (pd.notna(revenue[y]) and revenue[y] > 0 and
                    pd.notna(net_income[y])):
                npm[y] = (net_income[y] / revenue[y]) * 100
        ratios['Net Profit Margin %'] = npm

        if 'EBITDA' in is_.index:
            ebitda = is_.loc['EBITDA']
            ebitda_m = pd.Series(np.nan, index=years)
            for y in years:
                if pd.notna(revenue[y]) and revenue[y] > 0:
                    ebitda_m[y] = (ebitda[y] / revenue[y]) * 100
            ratios['EBITDA Margin %'] = ebitda_m

        # Growth
        ratios['Revenue Growth %'] = revenue.pct_change() * 100
        ratios['Net Income Growth %'] = net_income.pct_change() * 100
        if 'Net Operating Assets' in bs.index:
            ratios['NOA Growth %'] = (
                bs.loc['Net Operating Assets'].pct_change() * 100
            )
        if 'NOPAT' in is_.index:
            ratios['NOPAT Growth %'] = nopat.pct_change() * 100

        # Liquidity
        ca = self._get('Current Assets', True)
        cl = self._get('Current Liabilities', True)
        ratios['Current Ratio'] = pd.Series(
            [ca[y] / cl[y] if pd.notna(cl[y]) and cl[y] > 0 else np.nan
             for y in years],
            index=years
        )

        # Quick ratio (NEW)
        inv = self._get('Inventory', True)
        ratios['Quick Ratio'] = pd.Series(
            [(ca[y] - inv[y]) / cl[y]
             if pd.notna(cl[y]) and cl[y] > 0 else np.nan
             for y in years],
            index=years
        )

        # Interest Coverage
        ebit_vals = (
            is_.loc['EBIT'] if 'EBIT' in is_.index
            else pd.Series(np.nan, index=years)
        )
        fin_exp = (
            is_.loc['Interest Expense'] if 'Interest Expense' in is_.index
            else pd.Series(0, index=years)
        )
        icr = pd.Series(np.nan, index=years)
        for y in years:
            if pd.notna(ebit_vals[y]):
                if pd.notna(fin_exp[y]) and fin_exp[y] > 0.01:
                    icr[y] = min(ebit_vals[y] / fin_exp[y], 999)
                elif total_debt[y] <= 10 and ebit_vals[y] > 0:
                    icr[y] = 999
        ratios['Interest Coverage'] = icr

        # Debt Ratios
        ce = (
            bs.loc['Common Equity'] if 'Common Equity' in bs.index
            else pd.Series(np.nan, index=years)
        )
        ratios['Debt to Equity'] = pd.Series(
            [total_debt[y] / ce[y]
             if pd.notna(ce[y]) and ce[y] > 0 else np.nan
             for y in years],
            index=years
        )

        # Net Debt / EBITDA (NEW)
        if 'Net Debt' in bs.index and 'EBITDA' in is_.index:
            nd = bs.loc['Net Debt']
            ebitda_v = is_.loc['EBITDA']
            ratios['Net Debt / EBITDA'] = pd.Series(
                [nd[y] / ebitda_v[y]
                 if pd.notna(ebitda_v[y]) and ebitda_v[y] > 0 else np.nan
                 for y in years],
                index=years
            )

        # Sustainable Growth Rate (NEW)
        sgr = pd.Series(np.nan, index=years)
        for y in years:
            if pd.notna(roe[y]):
                # Assume retention ratio from data if DPS available
                if 'EPS Basic' in is_.index and 'Dividend Per Share' in is_.index:
                    eps_v = is_.loc['EPS Basic'][y]
                    dps_v = is_.loc['Dividend Per Share'][y]
                    if pd.notna(eps_v) and eps_v > 0:
                        retention = 1 - (dps_v / eps_v if pd.notna(dps_v) else 0)
                        sgr[y] = roe[y] * retention
                        continue
                # Default 70% retention
                sgr[y] = roe[y] * 0.70
        ratios['Sustainable Growth Rate %'] = sgr

        result = ratios.T
        self.log.info(
            f"Calculated {len(result)} ratios across {len(years)} years"
        )
        return result

    def calculate_fcf(self) -> pd.DataFrame:
        cols = self._data.columns
        fcf = pd.DataFrame(index=cols, dtype=float)

        ocf = self._get('Operating Cash Flow', True)
        capex = self._get('Capital Expenditure', True).abs()

        fcf['Operating Cash Flow'] = ocf
        fcf['Capital Expenditure'] = capex
        fcf['Free Cash Flow to Firm'] = ocf - capex

        # FCFE (NEW)
        ie = self._get('Interest Expense', True)
        debt_repay = (
            self._get('Debt Repayment', True)
            if self._has('Debt Repayment')
            else pd.Series(0, index=cols)
        )
        debt_proc = (
            self._get('Debt Proceeds', True)
            if self._has('Debt Proceeds')
            else pd.Series(0, index=cols)
        )
        fcfe = ocf - capex - ie + (debt_proc - debt_repay)
        fcf['Free Cash Flow to Equity'] = fcfe

        bs = self.reformulate_balance_sheet()
        if 'Total Assets' in bs.index:
            ta = bs.loc['Total Assets']
            fcf['FCF Yield %'] = (
                (ocf - capex) / ta.replace(0, np.nan) * 100
            )

        # Investing + Financing CFs
        if self._has('Investing Cash Flow'):
            fcf['Investing Cash Flow'] = self._get(
                'Investing Cash Flow', True
            )
        if self._has('Financing Cash Flow'):
            fcf['Financing Cash Flow'] = self._get(
                'Financing Cash Flow', True
            )
        if self._has('Dividends Paid'):
            fcf['Dividends Paid'] = self._get('Dividends Paid', True)

        return fcf.T

    def calculate_value_drivers(self) -> pd.DataFrame:
        cols = self._data.columns
        drivers = pd.DataFrame(index=cols, dtype=float)

        rev = self._get('Revenue')
        drivers['Revenue'] = rev
        drivers['Revenue Growth %'] = rev.pct_change() * 100

        is_ = self.reformulate_income_statement()
        if 'NOPAT' in is_.index:
            nopat = is_.loc['NOPAT']
            drivers['NOPAT'] = nopat
            drivers['NOPAT Margin %'] = (
                nopat / rev.replace(0, np.nan) * 100
            )

        bs = self.reformulate_balance_sheet()
        if 'Net Operating Assets' in bs.index:
            noa = bs.loc['Net Operating Assets']
            drivers['NOA'] = noa
            drivers['NOA Growth %'] = noa.pct_change() * 100

        if 'Invested Capital' in bs.index:
            ic = bs.loc['Invested Capital']
            drivers['Invested Capital'] = ic

        if 'Net Income (Reported)' in is_.index:
            ni = is_.loc['Net Income (Reported)']
            drivers['Net Income'] = ni
            drivers['Net Income Growth %'] = ni.pct_change() * 100

        # Operating Leverage (NEW)
        if 'EBIT' in is_.index:
            ebit = is_.loc['EBIT']
            ebit_growth = ebit.pct_change() * 100
            rev_growth = rev.pct_change() * 100
            dol = ebit_growth / rev_growth.replace(0, np.nan)
            drivers['Degree of Operating Leverage'] = dol

        return drivers.T

    def calculate_residual_income(
        self, cost_of_equity: float = 0.12
    ) -> pd.DataFrame:
        """NEW: Residual Income / Economic Value Added."""
        cols = self._data.columns
        ri = pd.DataFrame(index=cols, dtype=float)

        bs = self.reformulate_balance_sheet()
        is_ = self.reformulate_income_statement()

        nopat = (
            is_.loc['NOPAT'] if 'NOPAT' in is_.index
            else pd.Series(np.nan, index=cols)
        )
        ce = (
            bs.loc['Common Equity'] if 'Common Equity' in bs.index
            else pd.Series(np.nan, index=cols)
        )
        noa = (
            bs.loc['Net Operating Assets']
            if 'Net Operating Assets' in bs.index
            else pd.Series(np.nan, index=cols)
        )
        ni = (
            is_.loc['Net Income (Reported)']
            if 'Net Income (Reported)' in is_.index
            else pd.Series(np.nan, index=cols)
        )

        # Residual Income = NI - (Cost of Equity × Beginning Equity)
        ri_vals = pd.Series(np.nan, index=cols)
        for i, y in enumerate(cols):
            if i == 0:
                continue
            beg_ce = ce[cols[i - 1]]
            if pd.notna(ni[y]) and pd.notna(beg_ce) and beg_ce > 0:
                ri_vals[y] = ni[y] - (cost_of_equity * beg_ce)
        ri['Residual Income'] = ri_vals

        # EVA = NOPAT - (WACC × Invested Capital)
        wacc = cost_of_equity * 0.7 + 0.09 * 0.3 * (1 - 0.25)
        eva_vals = pd.Series(np.nan, index=cols)
        for i, y in enumerate(cols):
            if i == 0:
                continue
            beg_noa = noa[cols[i - 1]]
            if pd.notna(nopat[y]) and pd.notna(beg_noa) and beg_noa > 0:
                eva_vals[y] = nopat[y] - (wacc * beg_noa)
        ri['Economic Value Added (EVA)'] = eva_vals

        ri['Cost of Equity Used'] = cost_of_equity
        ri['WACC Used'] = wacc

        return ri.T

    def calculate_all(self) -> Dict[str, Any]:
        try:
            return {
                'reformulated_balance_sheet': self.reformulate_balance_sheet(),
                'reformulated_income_statement': self.reformulate_income_statement(),
                'ratios': self.calculate_ratios(),
                'free_cash_flow': self.calculate_fcf(),
                'value_drivers': self.calculate_value_drivers(),
                'residual_income': self.calculate_residual_income(),
                'quality_score': self._mapping_quality(),
            }
        except Exception as e:
            self.log.error(f"PN analysis failed: {e}", exc_info=True)
            return {'error': str(e)}

    def _mapping_quality(self) -> float:
        critical = [
            'Total Assets', 'Total Equity', 'Revenue', 'Net Income',
            'Income Before Tax', 'Interest Expense', 'Tax Expense',
            'Current Assets', 'Current Liabilities',
        ]
        found = sum(1 for m in critical if self._has(m))
        return (found / len(critical)) * 100

    def generate_insights(self) -> List[str]:
        insights = []
        try:
            ratios = self.calculate_ratios()
            if ratios.empty:
                return ["⚠️ No ratios calculated"]

            latest = ratios.columns[-1]

            if 'RNOA %' in ratios.index:
                v = ratios.loc['RNOA %', latest]
                if pd.notna(v):
                    label = (
                        "🚀 Excellent" if v > 20
                        else "✅ Good" if v > 10
                        else "⚠️ Weak" if v > 0
                        else "❌ Negative"
                    )
                    insights.append(f"{label} RNOA: {v:.1f}%")

            if 'Spread %' in ratios.index:
                v = ratios.loc['Spread %', latest]
                if pd.notna(v):
                    if v > 0:
                        insights.append(
                            f"✅ Positive spread ({v:.1f}%) — "
                            f"leverage creates value"
                        )
                    else:
                        insights.append(
                            f"❌ Negative spread ({v:.1f}%) — "
                            f"leverage destroys value"
                        )

            if 'Financial Leverage (FLEV)' in ratios.index:
                v = ratios.loc['Financial Leverage (FLEV)', latest]
                if pd.notna(v) and v < 0:
                    insights.append(
                        f"💰 Net cash position (FLEV: {v:.2f})"
                    )
                elif pd.notna(v) and v > 2:
                    insights.append(
                        f"⚠️ High leverage (FLEV: {v:.2f})"
                    )

            if all(r in ratios.index for r in ['ROE %', 'RNOA %']):
                roe_v = ratios.loc['ROE %', latest]
                rnoa_v = ratios.loc['RNOA %', latest]
                if pd.notna(roe_v) and pd.notna(rnoa_v):
                    lev_effect = roe_v - rnoa_v
                    if abs(rnoa_v) > abs(lev_effect):
                        insights.append(
                            "💡 ROE driven primarily by operations"
                        )
                    else:
                        insights.append(
                            "💡 ROE significantly influenced by leverage"
                        )

            # ROIC insight (NEW)
            if 'ROIC %' in ratios.index:
                roic = ratios.loc['ROIC %', latest]
                if pd.notna(roic):
                    if roic > Cfg.BENCHMARKS['cost_of_equity']:
                        insights.append(
                            f"✅ ROIC ({roic:.1f}%) exceeds cost of "
                            f"capital — value creation"
                        )
                    else:
                        insights.append(
                            f"⚠️ ROIC ({roic:.1f}%) below cost of "
                            f"capital — value destruction"
                        )

            # Sustainable growth (NEW)
            if 'Sustainable Growth Rate %' in ratios.index:
                sgr = ratios.loc['Sustainable Growth Rate %', latest]
                if pd.notna(sgr):
                    if 'Revenue Growth %' in ratios.index:
                        rev_g = ratios.loc['Revenue Growth %', latest]
                        if pd.notna(rev_g) and rev_g > sgr * 1.5:
                            insights.append(
                                f"⚠️ Revenue growth ({rev_g:.1f}%) "
                                f"exceeds sustainable rate ({sgr:.1f}%) "
                                f"— may need external financing"
                            )

            # Net Debt (NEW)
            if 'Net Debt / EBITDA' in ratios.index:
                nd_ebitda = ratios.loc['Net Debt / EBITDA', latest]
                if pd.notna(nd_ebitda):
                    if nd_ebitda < 0:
                        insights.append(
                            "💰 Net cash position (negative Net Debt)"
                        )
                    elif nd_ebitda > 3:
                        insights.append(
                            f"⚠️ High Net Debt/EBITDA: {nd_ebitda:.1f}x"
                        )

        except Exception as e:
            insights.append(f"⚠️ Insight generation error: {e}")
        return insights


# ═══════════════════════════════════════════════════════════
# SECTION 13: ADVANCED SCORING MODELS (NEW)
# ═══════════════════════════════════════════════════════════

class AltmanZScore:
    """Altman Z-Score for bankruptcy prediction."""

    @staticmethod
    def calculate(
        working_capital: float, total_assets: float,
        retained_earnings: float, ebit: float,
        market_cap_or_equity: float, total_liabilities: float,
        revenue: float
    ) -> Dict[str, Any]:
        if total_assets <= 0:
            return {'score': np.nan, 'zone': 'N/A', 'components': {}}

        ta = total_assets
        x1 = working_capital / ta
        x2 = retained_earnings / ta
        x3 = ebit / ta
        x4 = market_cap_or_equity / max(total_liabilities, 1)
        x5 = revenue / ta

        z = 1.2 * x1 + 1.4 * x2 + 3.3 * x3 + 0.6 * x4 + 1.0 * x5

        if z > 2.99:
            zone = 'Safe'
        elif z > 1.81:
            zone = 'Grey'
        else:
            zone = 'Distress'

        return {
            'score': round(z, 2),
            'zone': zone,
            'components': {
                'X1 (WC/TA)': round(x1, 4),
                'X2 (RE/TA)': round(x2, 4),
                'X3 (EBIT/TA)': round(x3, 4),
                'X4 (Equity/TL)': round(x4, 4),
                'X5 (Sales/TA)': round(x5, 4),
            }
        }

    @classmethod
    def from_pn(
        cls, pn: PenmanNissimAnalyzer
    ) -> Dict[str, Dict]:
        """Calculate Z-Score across all years from PN data."""
        results = {}
        try:
            bs = pn.reformulate_balance_sheet()
            is_ = pn.reformulate_income_statement()
            years = bs.columns

            for y in years:
                try:
                    ta = bs.loc['Total Assets', y] if 'Total Assets' in bs.index else np.nan
                    wc = bs.loc['Net Working Capital', y] if 'Net Working Capital' in bs.index else 0
                    re = 0
                    if pn._has('Retained Earnings'):
                        re = pn._get('Retained Earnings', True)[y]
                    ebit = is_.loc['EBIT', y] if 'EBIT' in is_.index else 0
                    eq = bs.loc['Common Equity', y] if 'Common Equity' in bs.index else 0
                    tl = bs.loc['Total Liabilities', y] if 'Total Liabilities' in bs.index else 0
                    rev = is_.loc['Revenue', y] if 'Revenue' in is_.index else 0

                    if pd.notna(ta) and ta > 0:
                        results[y] = cls.calculate(wc, ta, re, ebit, eq, tl, rev)
                except Exception:
                    continue
        except Exception:
            pass
        return results


class PiotroskiFScore:
    """Piotroski F-Score (0-9) for financial health."""

    @staticmethod
    def calculate(pn: PenmanNissimAnalyzer) -> Dict[str, Any]:
        try:
            bs = pn.reformulate_balance_sheet()
            is_ = pn.reformulate_income_statement()
            ratios = pn.calculate_ratios()
            fcf = pn.calculate_fcf()
            years = bs.columns

            if len(years) < 2:
                return {'score': np.nan, 'details': {}, 'years': {}}

            results = {}
            for i in range(1, len(years)):
                y = years[i]
                py = years[i - 1]
                score = 0
                details = {}

                # 1. ROA > 0
                ni = is_.loc['Net Income (Reported)', y] if 'Net Income (Reported)' in is_.index else np.nan
                ta = bs.loc['Total Assets', y] if 'Total Assets' in bs.index else np.nan
                if pd.notna(ni) and pd.notna(ta) and ta > 0:
                    roa = ni / ta
                    flag = 1 if roa > 0 else 0
                    details['ROA > 0'] = flag
                    score += flag

                # 2. OCF > 0
                if 'Operating Cash Flow' in fcf.index:
                    ocf = fcf.loc['Operating Cash Flow', y]
                    flag = 1 if pd.notna(ocf) and ocf > 0 else 0
                    details['OCF > 0'] = flag
                    score += flag

                # 3. ROA improving
                if 'ROA %' in ratios.index:
                    roa_curr = ratios.loc['ROA %', y]
                    roa_prev = ratios.loc['ROA %', py]
                    if pd.notna(roa_curr) and pd.notna(roa_prev):
                        flag = 1 if roa_curr > roa_prev else 0
                        details['ROA Improving'] = flag
                        score += flag

                # 4. Quality: OCF > NI (accruals)
                if pd.notna(ni) and 'Operating Cash Flow' in fcf.index:
                    ocf = fcf.loc['Operating Cash Flow', y]
                    if pd.notna(ocf):
                        flag = 1 if ocf > ni else 0
                        details['OCF > NI (Quality)'] = flag
                        score += flag

                # 5. Leverage decreasing
                if 'Debt to Equity' in ratios.index:
                    de_curr = ratios.loc['Debt to Equity', y]
                    de_prev = ratios.loc['Debt to Equity', py]
                    if pd.notna(de_curr) and pd.notna(de_prev):
                        flag = 1 if de_curr <= de_prev else 0
                        details['Leverage Decreasing'] = flag
                        score += flag

                # 6. Current ratio improving
                if 'Current Ratio' in ratios.index:
                    cr_curr = ratios.loc['Current Ratio', y]
                    cr_prev = ratios.loc['Current Ratio', py]
                    if pd.notna(cr_curr) and pd.notna(cr_prev):
                        flag = 1 if cr_curr > cr_prev else 0
                        details['Liquidity Improving'] = flag
                        score += flag

                # 7. No dilution (share capital constant)
                if pn._has('Share Capital'):
                    sc_curr = pn._get('Share Capital')[y]
                    sc_prev = pn._get('Share Capital')[py]
                    if pd.notna(sc_curr) and pd.notna(sc_prev):
                        flag = 1 if sc_curr <= sc_prev else 0
                        details['No Dilution'] = flag
                        score += flag

                # 8. Gross margin improving
                if 'Gross Profit Margin %' in ratios.index:
                    gm_curr = ratios.loc['Gross Profit Margin %', y]
                    gm_prev = ratios.loc['Gross Profit Margin %', py]
                    if pd.notna(gm_curr) and pd.notna(gm_prev):
                        flag = 1 if gm_curr > gm_prev else 0
                        details['Margin Improving'] = flag
                        score += flag

                # 9. Asset turnover improving
                if 'Net Operating Asset Turnover (NOAT)' in ratios.index:
                    at_curr = ratios.loc['Net Operating Asset Turnover (NOAT)', y]
                    at_prev = ratios.loc['Net Operating Asset Turnover (NOAT)', py]
                    if pd.notna(at_curr) and pd.notna(at_prev):
                        flag = 1 if at_curr > at_prev else 0
                        details['Turnover Improving'] = flag
                        score += flag

                results[y] = {'score': score, 'details': details}

            return results

        except Exception as e:
            return {'error': str(e)}


class ScenarioAnalyzer:
    """NEW: Scenario analysis for key metrics."""

    @staticmethod
    def run_scenarios(
        pn: PenmanNissimAnalyzer,
        revenue_scenarios: Dict[str, float] = None,
        margin_scenarios: Dict[str, float] = None,
    ) -> Dict[str, pd.DataFrame]:
        if revenue_scenarios is None:
            revenue_scenarios = {
                'Bear': -0.10, 'Base': 0.05, 'Bull': 0.15
            }
        if margin_scenarios is None:
            margin_scenarios = {
                'Bear': -0.02, 'Base': 0.0, 'Bull': 0.02
            }

        results = {}
        is_ = pn.reformulate_income_statement()
        bs = pn.reformulate_balance_sheet()
        latest = is_.columns[-1]

        rev = is_.loc['Revenue', latest] if 'Revenue' in is_.index else 0
        opm = 0
        if 'NOPAT' in is_.index and rev > 0:
            opm = is_.loc['NOPAT', latest] / rev

        for scenario, rev_change in revenue_scenarios.items():
            margin_adj = margin_scenarios.get(scenario, 0)
            proj_rev = rev * (1 + rev_change)
            proj_margin = opm + margin_adj
            proj_nopat = proj_rev * proj_margin

            noa = (
                bs.loc['Net Operating Assets', latest]
                if 'Net Operating Assets' in bs.index else 0
            )
            proj_rnoa = (
                (proj_nopat / noa * 100) if noa > 0 else np.nan
            )

            results[scenario] = pd.DataFrame({
                'Revenue': [proj_rev],
                'NOPAT': [proj_nopat],
                'NOPAT Margin %': [proj_margin * 100],
                'Projected RNOA %': [proj_rnoa],
            }, index=[f"FY{int(latest[:4]) + 1}"])

        return results


# ═══════════════════════════════════════════════════════════
# SECTION 14: ML FORECASTING (ENHANCED)
# ═══════════════════════════════════════════════════════════

class Forecaster:
    @staticmethod
    def forecast(
        series: pd.Series, periods: int = 3,
        model: str = 'auto'
    ) -> Dict:
        s = series.dropna()
        if len(s) < 3:
            return {'error': 'Insufficient data (need ≥3 points)'}

        X = np.arange(len(s)).reshape(-1, 1)
        y = s.values

        models = {
            'linear': make_pipeline(LinearRegression()),
            'polynomial': make_pipeline(
                PolynomialFeatures(2), LinearRegression()
            ),
        }

        # Add exponential smoothing if available (NEW)
        best_model_name = 'linear'

        if model != 'auto':
            best_model = models.get(model, models['linear'])
            best_model.fit(X, y)
            best_model_name = model
        else:
            best_model, best_mse = None, float('inf')
            split = max(1, len(s) // 5)
            for name, mdl in models.items():
                try:
                    mdl.fit(X[:-split], y[:-split])
                    pred = mdl.predict(X[-split:])
                    mse = np.mean((y[-split:] - pred) ** 2)
                    if mse < best_mse:
                        best_mse = mse
                        best_model = mdl
                        best_model_name = name
                except Exception:
                    continue
            if best_model is None:
                best_model = models['linear']
            best_model.fit(X, y)

        future_X = np.arange(len(s), len(s) + periods).reshape(-1, 1)
        preds = best_model.predict(future_X)

        residuals = y - best_model.predict(X)
        std = np.std(residuals)
        z = 1.96

        try:
            last_year = int(str(s.index[-1])[:4])
            future_labels = [str(last_year + i + 1) for i in range(periods)]
        except (ValueError, IndexError):
            future_labels = [f"T+{i + 1}" for i in range(periods)]

        # Exponential smoothing forecast (NEW)
        es_forecast = None
        if STATSMODELS_OK and len(s) >= 4:
            try:
                es_model = ExponentialSmoothing(
                    s.values, trend='add', seasonal=None
                ).fit(optimized=True)
                es_preds = es_model.forecast(periods)
                es_forecast = es_preds.tolist()
            except Exception:
                pass

        return {
            'periods': future_labels,
            'values': preds.tolist(),
            'lower': (preds - z * std).tolist(),
            'upper': (preds + z * std).tolist(),
            'model_used': best_model_name,
            'exponential_smoothing': es_forecast,
            'accuracy': {
                'rmse': float(np.sqrt(np.mean(residuals ** 2))),
                'mae': float(np.mean(np.abs(residuals))),
                'mape': float(
                    np.mean(np.abs(residuals / np.where(
                        np.abs(y) > EPS, y, 1
                    ))) * 100
                ),
            }
        }

    @staticmethod
    def forecast_multiple(
        df: pd.DataFrame, periods: int = 3,
        metrics: Optional[List[str]] = None
    ) -> Dict:
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
# SECTION 15: VISUALIZATION FACTORY (NEW)
# ═══════════════════════════════════════════════════════════

class ChartFactory:
    """Centralised chart generation — no scattered Plotly code."""

    @staticmethod
    def radar_chart(
        categories: List[str], values: List[float],
        title: str = '', benchmark: Optional[List[float]] = None
    ) -> go.Figure:
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=values + [values[0]],
            theta=categories + [categories[0]],
            fill='toself', name='Company',
            line=dict(color='royalblue', width=2)
        ))
        if benchmark:
            fig.add_trace(go.Scatterpolar(
                r=benchmark + [benchmark[0]],
                theta=categories + [categories[0]],
                fill='toself', name='Benchmark',
                line=dict(color='gray', width=1, dash='dash'),
                opacity=0.5
            ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True)),
            title=title, height=450
        )
        return fig

    @staticmethod
    def waterfall_chart(
        labels: List[str], values: List[float],
        title: str = ''
    ) -> go.Figure:
        measures = ['relative'] * len(labels)
        measures[0] = 'absolute'
        measures[-1] = 'total'
        fig = go.Figure(go.Waterfall(
            name='', orientation='v',
            measure=measures,
            x=labels, y=values,
            connector=dict(line=dict(color='rgb(63,63,63)')),
            increasing=dict(marker=dict(color='green')),
            decreasing=dict(marker=dict(color='red')),
            totals=dict(marker=dict(color='blue')),
        ))
        fig.update_layout(title=title, height=400)
        return fig

    @staticmethod
    def gauge_chart(
        value: float, title: str = '',
        ranges: Optional[Dict] = None
    ) -> go.Figure:
        if ranges is None:
            ranges = {
                'red': [0, 30], 'yellow': [30, 60], 'green': [60, 100]
            }
        fig = go.Figure(go.Indicator(
            mode='gauge+number+delta',
            value=value,
            title={'text': title},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': 'darkblue'},
                'steps': [
                    {'range': ranges['red'], 'color': '#ff6b6b'},
                    {'range': ranges['yellow'], 'color': '#ffd93d'},
                    {'range': ranges['green'], 'color': '#6bcb77'},
                ],
            }
        ))
        fig.update_layout(height=300)
        return fig

    @staticmethod
    def heatmap(
        df: pd.DataFrame, title: str = ''
    ) -> go.Figure:
        # Truncate long labels
        labels_x = [str(c)[:20] for c in df.columns]
        labels_y = [str(i)[:30] for i in df.index]
        fig = go.Figure(go.Heatmap(
            z=df.values,
            x=labels_x, y=labels_y,
            colorscale='RdYlGn',
            text=np.round(df.values, 2),
            texttemplate='%{text}',
            textfont={'size': 9},
        ))
        fig.update_layout(
            title=title, height=max(400, len(df) * 25)
        )
        return fig

    @staticmethod
    def multi_line(
        df: pd.DataFrame, title: str = '',
        y_label: str = ''
    ) -> go.Figure:
        fig = go.Figure()
        for idx in df.index:
            vals = df.loc[idx]
            fig.add_trace(go.Scatter(
                x=df.columns.tolist(), y=vals.values.tolist(),
                mode='lines+markers', name=str(idx)[:30]
            ))
        fig.update_layout(
            title=title, height=400,
            yaxis_title=y_label, hovermode='x unified'
        )
        return fig

    @staticmethod
    def stacked_bar(
        df: pd.DataFrame, title: str = ''
    ) -> go.Figure:
        fig = go.Figure()
        colors = px.colors.qualitative.Set2
        for i, idx in enumerate(df.index):
            fig.add_trace(go.Bar(
                x=df.columns.tolist(),
                y=df.loc[idx].values.tolist(),
                name=str(idx)[:25],
                marker_color=colors[i % len(colors)]
            ))
        fig.update_layout(
            barmode='stack', title=title,
            height=400, hovermode='x unified'
        )
        return fig

    @staticmethod
    def roe_decomposition_waterfall(
        rnoa: float, flev: float, spread: float, roe: float,
        year: str = ''
    ) -> go.Figure:
        lev_effect = flev * spread
        labels = ['RNOA', 'Leverage Effect', 'ROE']
        values = [rnoa, lev_effect, roe]
        return ChartFactory.waterfall_chart(
            labels, values,
            title=f"ROE Decomposition {year}"
        )

    @staticmethod
    def dupont_tree(
        npm: float, at: float, em: float,
        title: str = 'DuPont 3-Factor'
    ) -> go.Figure:
        """Simple DuPont visualization as grouped bars."""
        fig = go.Figure()
        factors = ['Net Profit Margin', 'Asset Turnover', 'Equity Multiplier']
        values = [npm * 100, at, em]
        colors = ['#2196F3', '#4CAF50', '#FF9800']
        fig.add_trace(go.Bar(
            x=factors, y=values,
            marker_color=colors,
            text=[f'{v:.2f}' for v in values],
            textposition='outside'
        ))
        fig.update_layout(title=title, height=350)
        return fig


# ═══════════════════════════════════════════════════════════
# SECTION 16: NUMBER FORMATTING & EXPORT (ENHANCED)
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
    if a >= 1:
        return f"{sign}₹{a:.0f}"
    if a > 0:
        return f"{sign}₹{a:.2f}"
    return "₹0"


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
    if a >= 1:
        return f"{sign}${a:.0f}"
    if a > 0:
        return f"{sign}${a:.2f}"
    return "$0"


@lru_cache(maxsize=2)
def get_formatter(fmt: str) -> Callable:
    return fmt_indian if fmt == 'Indian' else fmt_intl


class ExportManager:
    """Enhanced with JSON, PDF-via-HTML, and PN results support."""

    @staticmethod
    def to_excel(
        analysis: Dict, company: str = 'Analysis',
        pn_results: Optional[Dict] = None
    ) -> bytes:
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine='xlsxwriter') as w:
            if 'summary' in analysis:
                pd.DataFrame([analysis['summary']]).to_excel(
                    w, 'Summary', index=False
                )
            for cat, df in analysis.get('ratios', {}).items():
                if isinstance(df, pd.DataFrame) and not df.empty:
                    df.to_excel(w, cat[:31])
            if 'insights' in analysis:
                pd.DataFrame({'Insights': analysis['insights']}).to_excel(
                    w, 'Insights', index=False
                )
            # PN results (NEW)
            if pn_results:
                for key in [
                    'reformulated_balance_sheet',
                    'reformulated_income_statement',
                    'ratios', 'free_cash_flow', 'value_drivers',
                    'residual_income',
                ]:
                    df = pn_results.get(key)
                    if isinstance(df, pd.DataFrame) and not df.empty:
                        sheet = key.replace('_', ' ').title()[:31]
                        df.to_excel(w, sheet)
        buf.seek(0)
        return buf.read()

    @staticmethod
    def to_json(
        analysis: Dict, company: str = 'Analysis'
    ) -> str:
        """NEW: Export to JSON."""
        export = {
            'company': company,
            'generated': datetime.now().isoformat(),
            'version': Cfg.VERSION,
        }
        for key, val in analysis.items():
            if isinstance(val, pd.DataFrame):
                export[key] = val.to_dict()
            elif isinstance(val, dict):
                clean = {}
                for k, v in val.items():
                    if isinstance(v, pd.DataFrame):
                        clean[k] = v.to_dict()
                    elif isinstance(v, (np.floating, np.integer)):
                        clean[k] = float(v)
                    else:
                        clean[k] = v
                export[key] = clean
            elif isinstance(val, (list, str, int, float, bool)):
                export[key] = val
        return json.dumps(export, indent=2, default=str)

    @staticmethod
    def to_markdown(
        analysis: Dict, company: str = 'Analysis'
    ) -> str:
        lines = [
            f"# {company} Financial Analysis Report",
            f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"Platform: Elite Financial Analytics v{Cfg.VERSION}",
            "\n---\n",
            "## Summary\n",
        ]
        s = analysis.get('summary', {})
        lines.extend([f"- **{k}**: {v}" for k, v in s.items()])
        if 'insights' in analysis:
            lines.append("\n## Key Insights\n")
            lines.extend([f"- {i}" for i in analysis['insights']])
        # Ratios
        for cat, df in analysis.get('ratios', {}).items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                lines.append(f"\n## {cat} Ratios\n")
                lines.append(df.to_markdown())
        return '\n'.join(lines)

    @staticmethod
    def to_html_report(
        analysis: Dict, company: str = 'Analysis',
        pn_results: Optional[Dict] = None
    ) -> str:
        """NEW: Generate HTML report (can be printed as PDF)."""
        html = f"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>{company} Financial Report</title>
<style>
body {{font-family: 'Segoe UI', sans-serif; max-width: 1200px; margin: auto; padding: 20px}}
h1 {{color: #1a237e; border-bottom: 3px solid #3f51b5}}
h2 {{color: #283593; margin-top: 30px}}
table {{border-collapse: collapse; width: 100%; margin: 15px 0}}
th, td {{border: 1px solid #ddd; padding: 8px; text-align: right}}
th {{background: #e8eaf6; font-weight: 600}}
td:first-child {{text-align: left}}
.insight {{padding: 10px; margin: 5px 0; border-radius: 5px; border-left: 4px solid}}
.positive {{background: #e8f5e9; border-color: #4caf50}}
.warning {{background: #fff3e0; border-color: #ff9800}}
.negative {{background: #ffebee; border-color: #f44336}}
.footer {{color: #666; font-size: 0.8em; margin-top: 40px; border-top: 1px solid #ddd; padding-top: 10px}}
@media print {{body {{font-size: 10pt}}}}
</style>
</head><body>
<h1>📊 {company} Financial Analysis Report</h1>
<p><em>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')} | 
Elite Financial Analytics v{Cfg.VERSION}</em></p>
"""
        # Summary
        s = analysis.get('summary', {})
        html += "<h2>Summary</h2><table>"
        for k, v in s.items():
            if not isinstance(v, dict):
                html += f"<tr><td><b>{k}</b></td><td>{v}</td></tr>"
        html += "</table>"

        # Insights
        insights = analysis.get('insights', [])
        if insights:
            html += "<h2>Key Insights</h2>"
            for i in insights:
                css = (
                    'positive' if any(c in i for c in ['✅', '🚀', '💰'])
                    else 'negative' if any(c in i for c in ['❌', '📉'])
                    else 'warning'
                )
                html += f'<div class="insight {css}">{i}</div>'

        # Ratios
        for cat, df in analysis.get('ratios', {}).items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                html += f"<h2>{cat} Ratios</h2>"
                html += df.to_html(float_format='%.2f', na_rep='-')

        # PN Results
        if pn_results:
            for key in ['ratios', 'free_cash_flow']:
                df = pn_results.get(key)
                if isinstance(df, pd.DataFrame) and not df.empty:
                    title = key.replace('_', ' ').title()
                    html += f"<h2>Penman-Nissim: {title}</h2>"
                    html += df.to_html(float_format='%.2f', na_rep='-')

        html += f"""
<div class="footer">
Report generated by Elite Financial Analytics Platform v{Cfg.VERSION}<br>
Methodology: Penman-Nissim Framework (Nissim & Penman, 2001)
</div>
</body></html>"""
        return html


# ═══════════════════════════════════════════════════════════
# SECTION 17: SAMPLE DATA (FIXED + EXPANDED)
# ═══════════════════════════════════════════════════════════

class SampleData:
    """Multiple sample datasets. FIXED: proper DataFrame construction."""

    @staticmethod
    def _build(data: Dict[str, List], years: List[str]) -> pd.DataFrame:
        """Correct DataFrame construction: metrics as rows, years as cols."""
        df = pd.DataFrame.from_dict(data, orient='index', columns=years)
        df = df.apply(pd.to_numeric, errors='coerce')
        return df

    @classmethod
    def indian_tech(cls) -> Tuple[pd.DataFrame, str]:
        years = ['201903', '202003', '202103', '202203', '202303']
        data = {
            'BalanceSheet::Total Assets': [45000, 52000, 61000, 72000, 85000],
            'BalanceSheet::Current Assets': [28000, 32000, 38000, 45000, 53000],
            'BalanceSheet::Cash and Cash Equivalents': [12000, 14000, 17000, 21000, 25000],
            'BalanceSheet::Current Investments': [2000, 2500, 3000, 3500, 4000],
            'BalanceSheet::Inventories': [2000, 2300, 2700, 3200, 3800],
            'BalanceSheet::Trade Receivables': [8000, 9200, 10800, 12700, 15000],
            'BalanceSheet::Other Current Assets': [4000, 4000, 4500, 4600, 5200],
            'BalanceSheet::Property Plant and Equipment': [10000, 12000, 14000, 16500, 19500],
            'BalanceSheet::Goodwill': [1500, 1500, 2000, 2000, 2500],
            'BalanceSheet::Intangible Assets': [800, 900, 1000, 1200, 1500],
            'BalanceSheet::Capital Work in Progress': [700, 600, 1000, 1300, 1500],
            'BalanceSheet::Non-Current Investments': [2000, 3000, 3000, 4000, 5000],
            'BalanceSheet::Deferred Tax Assets': [500, 600, 700, 800, 1000],
            'BalanceSheet::Total Equity': [27000, 32000, 38500, 46500, 56000],
            'BalanceSheet::Share Capital': [1000, 1000, 1000, 1000, 1000],
            'BalanceSheet::Reserves and Surplus': [26000, 31000, 37500, 45500, 55000],
            'BalanceSheet::Total Current Liabilities': [10000, 11000, 12500, 14000, 16000],
            'BalanceSheet::Trade Payables': [4000, 4400, 4900, 5500, 6200],
            'BalanceSheet::Other Current Liabilities': [3000, 3300, 3900, 4300, 5000],
            'BalanceSheet::Short Term Borrowings': [3000, 3300, 3700, 4200, 4800],
            'BalanceSheet::Long Term Borrowings': [6000, 6600, 7300, 8200, 9200],
            'BalanceSheet::Other Non-Current Liabilities': [1000, 1100, 1200, 1300, 1500],
            'BalanceSheet::Provisions': [1000, 1000, 1500, 2000, 2300],
            'ProfitLoss::Revenue From Operations': [35000, 38000, 45000, 54000, 65000],
            'ProfitLoss::Other Income': [400, 460, 530, 600, 700],
            'ProfitLoss::Cost of Materials Consumed': [21000, 22000, 25200, 29700, 35100],
            'ProfitLoss::Changes in Inventories': [-200, -300, -400, -500, -600],
            'ProfitLoss::Employee Benefit Expenses': [4000, 4400, 5150, 6075, 7150],
            'ProfitLoss::Depreciation and Amortisation Expenses': [1500, 1800, 2100, 2500, 3000],
            'ProfitLoss::Other Expenses': [2700, 3080, 3580, 4075, 4850],
            'ProfitLoss::Profit Before Exceptional Items and Tax': [6000, 7200, 9500, 12150, 15600],
            'ProfitLoss::Exceptional Items': [0, 0, -200, 0, 0],
            'ProfitLoss::Finance Cost': [800, 880, 970, 1090, 1220],
            'ProfitLoss::Profit Before Tax': [5600, 6780, 8860, 11660, 15080],
            'ProfitLoss::Tax Expense': [1680, 2034, 2658, 3498, 4524],
            'ProfitLoss::Profit After Tax': [3920, 4746, 6202, 8162, 10556],
            'ProfitLoss::Basic EPS': [39.2, 47.46, 62.02, 81.62, 105.56],
            'ProfitLoss::Diluted EPS': [38.5, 46.8, 61.2, 80.5, 104.2],
            'ProfitLoss::Dividend Per Share': [8, 10, 12, 15, 20],
            'CashFlow::Net Cash from Operating Activities': [5500, 6600, 8800, 11000, 14000],
            'CashFlow::Purchase of Fixed Assets': [-2800, -3200, -3800, -4500, -5300],
            'CashFlow::Purchase of Investments': [-1000, -1500, -500, -1500, -1500],
            'CashFlow::Sale of Investments': [500, 800, 600, 500, 1000],
            'CashFlow::Net Cash Used in Investing': [-3300, -3900, -3700, -5500, -5800],
            'CashFlow::Dividends Paid': [-800, -1000, -1200, -1500, -2000],
            'CashFlow::Repayment of Borrowings': [-500, -600, -700, -800, -900],
            'CashFlow::Proceeds from Borrowings': [800, 900, 1100, 1200, 1300],
            'CashFlow::Net Cash Used in Financing': [-500, -700, -800, -1100, -1600],
            'CashFlow::Net Increase in Cash': [1700, 2000, 4300, 4400, 6600],
        }
        return cls._build(data, years), "TechCorp India Ltd."

    @classmethod
    def indian_manufacturing(cls) -> Tuple[pd.DataFrame, str]:
        years = ['201903', '202003', '202103', '202203', '202303']
        data = {
            'BalanceSheet::Total Assets': [80000, 85000, 78000, 95000, 110000],
            'BalanceSheet::Current Assets': [35000, 37000, 32000, 42000, 50000],
            'BalanceSheet::Cash and Cash Equivalents': [5000, 4500, 6000, 7000, 8500],
            'BalanceSheet::Inventories': [12000, 13500, 10000, 15000, 18000],
            'BalanceSheet::Trade Receivables': [10000, 11000, 9000, 12000, 14000],
            'BalanceSheet::Other Current Assets': [8000, 8000, 7000, 8000, 9500],
            'BalanceSheet::Property Plant and Equipment': [30000, 32000, 30000, 35000, 40000],
            'BalanceSheet::Capital Work in Progress': [5000, 6000, 8000, 7000, 8000],
            'BalanceSheet::Intangible Assets': [500, 500, 500, 600, 700],
            'BalanceSheet::Non-Current Investments': [3000, 3000, 3000, 4000, 4000],
            'BalanceSheet::Deferred Tax Assets': [500, 500, 500, 400, 300],
            'BalanceSheet::Total Equity': [40000, 42000, 38000, 48000, 58000],
            'BalanceSheet::Share Capital': [5000, 5000, 5000, 5000, 5000],
            'BalanceSheet::Reserves and Surplus': [35000, 37000, 33000, 43000, 53000],
            'BalanceSheet::Total Current Liabilities': [18000, 20000, 22000, 22000, 24000],
            'BalanceSheet::Trade Payables': [8000, 9000, 7500, 10000, 11000],
            'BalanceSheet::Short Term Borrowings': [6000, 7000, 10000, 8000, 8000],
            'BalanceSheet::Other Current Liabilities': [4000, 4000, 4500, 4000, 5000],
            'BalanceSheet::Long Term Borrowings': [18000, 19000, 15000, 20000, 22000],
            'BalanceSheet::Other Non-Current Liabilities': [2000, 2000, 1500, 2500, 3000],
            'BalanceSheet::Provisions': [2000, 2000, 1500, 2500, 3000],
            'ProfitLoss::Revenue From Operations': [60000, 55000, 40000, 70000, 85000],
            'ProfitLoss::Other Income': [800, 700, 500, 900, 1100],
            'ProfitLoss::Cost of Materials Consumed': [36000, 33000, 24000, 42000, 51000],
            'ProfitLoss::Changes in Inventories': [500, -1500, 3500, -5000, -3000],
            'ProfitLoss::Employee Benefit Expenses': [7000, 7200, 6500, 8000, 9500],
            'ProfitLoss::Depreciation and Amortisation Expenses': [3500, 3800, 3500, 4000, 4500],
            'ProfitLoss::Other Expenses': [5000, 5200, 4000, 6000, 7000],
            'ProfitLoss::Finance Cost': [2200, 2400, 2000, 2500, 2800],
            'ProfitLoss::Profit Before Tax': [7600, 4600, 1000, 13400, 14300],
            'ProfitLoss::Tax Expense': [2280, 1380, 300, 4020, 4290],
            'ProfitLoss::Profit After Tax': [5320, 3220, 700, 9380, 10010],
            'ProfitLoss::Basic EPS': [10.64, 6.44, 1.40, 18.76, 20.02],
            'ProfitLoss::Dividend Per Share': [3, 2, 0, 5, 6],
            'CashFlow::Net Cash from Operating Activities': [8000, 6000, 5000, 13000, 15000],
            'CashFlow::Purchase of Fixed Assets': [-5000, -6000, -3000, -7000, -8000],
            'CashFlow::Net Cash Used in Investing': [-5500, -6500, -3500, -7500, -8500],
            'CashFlow::Net Cash Used in Financing': [-3000, -1000, 500, -4000, -5000],
        }
        return cls._build(data, years), "SteelForge Industries Ltd."

    @classmethod
    def indian_banking(cls) -> Tuple[pd.DataFrame, str]:
        years = ['201903', '202003', '202103', '202203', '202303']
        data = {
            'BalanceSheet::Total Assets': [500000, 550000, 620000, 700000, 800000],
            'BalanceSheet::Cash and Cash Equivalents': [30000, 33000, 40000, 45000, 50000],
            'BalanceSheet::Current Investments': [50000, 55000, 65000, 70000, 80000],
            'BalanceSheet::Trade Receivables': [300000, 330000, 370000, 420000, 480000],
            'BalanceSheet::Current Assets': [400000, 440000, 500000, 560000, 640000],
            'BalanceSheet::Property Plant and Equipment': [10000, 11000, 12000, 13000, 15000],
            'BalanceSheet::Non-Current Investments': [80000, 88000, 100000, 115000, 130000],
            'BalanceSheet::Total Equity': [50000, 55000, 62000, 72000, 85000],
            'BalanceSheet::Share Capital': [5000, 5000, 5000, 5000, 5000],
            'BalanceSheet::Reserves and Surplus': [45000, 50000, 57000, 67000, 80000],
            'BalanceSheet::Total Current Liabilities': [400000, 440000, 500000, 560000, 640000],
            'BalanceSheet::Short Term Borrowings': [350000, 385000, 435000, 490000, 560000],
            'BalanceSheet::Other Current Liabilities': [50000, 55000, 65000, 70000, 80000],
            'BalanceSheet::Long Term Borrowings': [40000, 44000, 48000, 55000, 60000],
            'BalanceSheet::Provisions': [10000, 11000, 10000, 13000, 15000],
            'ProfitLoss::Revenue From Operations': [40000, 42000, 38000, 45000, 55000],
            'ProfitLoss::Other Income': [5000, 5500, 6000, 7000, 8000],
            'ProfitLoss::Employee Benefit Expenses': [8000, 8500, 9000, 10000, 11000],
            'ProfitLoss::Other Expenses': [5000, 5500, 6000, 6500, 7000],
            'ProfitLoss::Depreciation and Amortisation Expenses': [1500, 1600, 1700, 1800, 2000],
            'ProfitLoss::Finance Cost': [25000, 26000, 22000, 24000, 30000],
            'ProfitLoss::Profit Before Tax': [5500, 5900, 5300, 9700, 13000],
            'ProfitLoss::Tax Expense': [1650, 1770, 1590, 2910, 3900],
            'ProfitLoss::Profit After Tax': [3850, 4130, 3710, 6790, 9100],
            'ProfitLoss::Basic EPS': [7.70, 8.26, 7.42, 13.58, 18.20],
            'ProfitLoss::Dividend Per Share': [2, 2, 1.5, 3, 4],
            'CashFlow::Net Cash from Operating Activities': [15000, 18000, 25000, 20000, 22000],
            'CashFlow::Purchase of Fixed Assets': [-2000, -2200, -2500, -2800, -3000],
            'CashFlow::Net Cash Used in Investing': [-10000, -12000, -15000, -18000, -20000],
            'CashFlow::Net Cash Used in Financing': [-2000, -3000, -3000, -5000, -5000],
        }
        return cls._build(data, years), "SafeBank Financial Ltd."

    @classmethod
    def indian_pharma(cls) -> Tuple[pd.DataFrame, str]:
        years = ['201903', '202003', '202103', '202203', '202303']
        data = {
            'BalanceSheet::Total Assets': [25000, 28000, 35000, 38000, 42000],
            'BalanceSheet::Current Assets': [14000, 16000, 20000, 22000, 25000],
            'BalanceSheet::Cash and Cash Equivalents': [3000, 4000, 7000, 6000, 7500],
            'BalanceSheet::Inventories': [4000, 4500, 5000, 6000, 7000],
            'BalanceSheet::Trade Receivables': [5000, 5500, 6000, 7000, 8000],
            'BalanceSheet::Other Current Assets': [2000, 2000, 2000, 3000, 2500],
            'BalanceSheet::Property Plant and Equipment': [6000, 7000, 8000, 9000, 10000],
            'BalanceSheet::Intangible Assets': [2000, 2000, 3000, 3000, 3500],
            'BalanceSheet::Goodwill': [1000, 1000, 2000, 2000, 2000],
            'BalanceSheet::Total Equity': [16000, 18500, 24000, 26000, 30000],
            'BalanceSheet::Share Capital': [500, 500, 500, 500, 500],
            'BalanceSheet::Reserves and Surplus': [15500, 18000, 23500, 25500, 29500],
            'BalanceSheet::Total Current Liabilities': [5000, 5500, 6000, 7000, 7000],
            'BalanceSheet::Trade Payables': [3000, 3200, 3500, 4000, 4200],
            'BalanceSheet::Short Term Borrowings': [1000, 1200, 1000, 1500, 1300],
            'BalanceSheet::Long Term Borrowings': [3000, 3000, 4000, 3500, 3500],
            'BalanceSheet::Other Non-Current Liabilities': [500, 500, 500, 500, 500],
            'BalanceSheet::Provisions': [500, 500, 500, 1000, 1000],
            'ProfitLoss::Revenue From Operations': [18000, 20000, 28000, 30000, 34000],
            'ProfitLoss::Other Income': [300, 350, 500, 400, 450],
            'ProfitLoss::Cost of Materials Consumed': [7200, 8000, 11200, 12000, 13600],
            'ProfitLoss::Employee Benefit Expenses': [3000, 3200, 3800, 4200, 4800],
            'ProfitLoss::Depreciation and Amortisation Expenses': [1200, 1400, 1600, 1800, 2000],
            'ProfitLoss::Other Expenses': [3500, 3800, 5000, 5500, 6200],
            'ProfitLoss::Finance Cost': [400, 380, 350, 400, 420],
            'ProfitLoss::Profit Before Tax': [3000, 3570, 6550, 6500, 7430],
            'ProfitLoss::Tax Expense': [750, 893, 1638, 1625, 1858],
            'ProfitLoss::Profit After Tax': [2250, 2678, 4913, 4875, 5573],
            'ProfitLoss::Basic EPS': [45.0, 53.55, 98.25, 97.50, 111.45],
            'ProfitLoss::Dividend Per Share': [10, 12, 20, 20, 25],
            'CashFlow::Net Cash from Operating Activities': [3500, 4000, 7000, 6000, 7500],
            'CashFlow::Purchase of Fixed Assets': [-1500, -2000, -2500, -2500, -3000],
            'CashFlow::Net Cash Used in Investing': [-2000, -2500, -4000, -3500, -3500],
            'CashFlow::Net Cash Used in Financing': [-800, -1000, -1500, -1500, -2000],
        }
        return cls._build(data, years), "PharmaLife Sciences Ltd."

    @classmethod
    def get_all_samples(cls) -> Dict[str, Callable]:
        return {
            '🖥️ Indian Tech Company': cls.indian_tech,
            '🏭 Indian Manufacturing': cls.indian_manufacturing,
            '🏦 Indian Banking': cls.indian_banking,
            '💊 Indian Pharma': cls.indian_pharma,
        }


# ═══════════════════════════════════════════════════════════
# SECTION 18: UI COMPONENTS
# ═══════════════════════════════════════════════════════════

class MappingEditor:
    """NEW: Interactive metric mapping editor."""

    @staticmethod
    def render(
        data: pd.DataFrame,
        existing_mappings: Optional[Dict[str, str]] = None
    ) -> Optional[Dict[str, str]]:
        MetricPatterns._build()
        targets_by_stmt = MetricPatterns.get_targets_by_statement()
        all_targets = MetricPatterns.get_all_targets()
        sources = list(data.index)
        mappings = dict(existing_mappings) if existing_mappings else {}

        st.subheader("📝 Mapping Editor")
        st.caption(
            "Map your data metrics to standard financial targets. "
            "Unmap by selecting '-- Not Mapped --'."
        )

        # Group sources by statement
        source_groups = defaultdict(list)
        for s in sources:
            parts = str(s).split('::')
            stmt = parts[0] if len(parts) > 1 else 'Other'
            source_groups[stmt].append(s)

        changed = False
        for stmt, src_list in sorted(source_groups.items()):
            with st.expander(
                f"{stmt} ({len(src_list)} metrics)", expanded=False
            ):
                relevant_targets = targets_by_stmt.get(stmt, all_targets)
                options = ['-- Not Mapped --'] + relevant_targets

                for source in src_list:
                    clean_name = (
                        source.split('::')[-1]
                        if '::' in source else source
                    )
                    current = mappings.get(source, '-- Not Mapped --')
                    idx = (
                        options.index(current)
                        if current in options else 0
                    )

                    new_val = st.selectbox(
                        clean_name, options, index=idx,
                        key=f"map_{source}"
                    )

                    if new_val != '-- Not Mapped --':
                        if source not in mappings or mappings[source] != new_val:
                            mappings[source] = new_val
                            changed = True
                    elif source in mappings:
                        del mappings[source]
                        changed = True

        if changed:
            # Check for duplicate target assignments
            target_counts = defaultdict(int)
            for t in mappings.values():
                target_counts[t] += 1
            duplicates = [t for t, c in target_counts.items() if c > 1]
            if duplicates:
                st.warning(
                    f"⚠️ Duplicate mappings: {', '.join(duplicates)}"
                )

        return mappings if mappings else None


# ═══════════════════════════════════════════════════════════
# SECTION 19: MAIN APPLICATION CLASS (ENHANCED)
# ═══════════════════════════════════════════════════════════

class App:
    def __init__(self):
        self._init_state()
        Cfg.from_session()
        self.log = Log.get('App')
        self.parser = CapitalineParser()
        self.cleaner = DataCleaner()
        self.analyzer = FinancialAnalyzer()
        self.mapper = MetricMapper()
        self.compressor = CompressedFileHandler()

        if not st.session_state.get('mapper_initialized'):
            self.mapper.initialize(Cfg.KAGGLE_URL)
            st.session_state.mapper_initialized = True

    def _init_state(self):
        defaults = {
            'data': None, 'data_hash': None,
            'company': 'Company',
            'mappings': None, 'pn_mappings': None,
            'pn_results': None,
            'number_format': 'Indian', 'debug_mode': False,
            'kaggle_api_url': '', 'mapper_initialized': False,
            'forecast_results': None,
            'detected_unit': None,
            'validation_report': None,
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
        .info-box {padding:15px;border-radius:8px;margin:10px 0;
            border-left:4px solid}
        .info-box.blue {background:#e3f2fd;border-color:#2196f3}
        .info-box.green {background:#e8f5e9;border-color:#4caf50}
        .info-box.orange {background:#fff3e0;border-color:#ff9800}
        </style>""", unsafe_allow_html=True)

    def _header(self):
        st.markdown(
            '<h1 class="main-title">💹 Elite Financial Analytics v7.0</h1>',
            unsafe_allow_html=True
        )
        c1, c2, c3, c4, c5 = st.columns(5)
        status = self.mapper.get_status()
        c1.metric("Version", Cfg.VERSION)
        c2.metric(
            "AI",
            "Kaggle GPU" if status['kaggle_ok']
            else "Local" if status['local_ok']
            else "Fuzzy"
        )
        c3.metric("Cache Hit", f"{status['cache_hit_rate']:.0f}%")
        c4.metric("Format", Cfg.NUMBER_FORMAT)
        data = st.session_state.data
        c5.metric(
            "Data",
            f"{len(data)} metrics" if data is not None else "None"
        )

    def _sidebar(self):
        st.sidebar.title("⚙️ Configuration")

        # Company name (NEW)
        company = st.sidebar.text_input(
            "Company Name",
            st.session_state.company,
            placeholder="Enter company name"
        )
        st.session_state.company = bleach.clean(
            company, tags=[], strip=True
        )

        # Kaggle
        with st.sidebar.expander("🖥️ Kaggle GPU"):
            url = st.text_input(
                "Ngrok URL", st.session_state.kaggle_api_url,
                placeholder="https://xxxx.ngrok-free.app"
            )
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
                "Upload Financial Statements",
                type=Cfg.ALLOWED_TYPES,
                accept_multiple_files=True
            )
            if files and st.sidebar.button("Process", type="primary"):
                self._process_files(files)
        else:
            samples = SampleData.get_all_samples()
            choice = st.sidebar.selectbox("Dataset", list(samples.keys()))
            if st.sidebar.button("Load Sample", type="primary"):
                df, name = samples[choice]()
                st.session_state.data = df
                st.session_state.company = name
                st.session_state.mappings = None
                st.session_state.pn_mappings = None
                st.session_state.pn_results = None
                st.sidebar.success(f"Loaded: {name}")

        # Settings
        st.sidebar.header("⚙️ Settings")
        fmt = st.sidebar.radio("Number Format", ["Indian", "International"])
        st.session_state.number_format = fmt
        Cfg.NUMBER_FORMAT = fmt

        # Unit scale (NEW)
        unit_opts = {
            'Absolute': 1, 'In Thousands': 1e3,
            'In Lakhs': 1e5, 'In Crores': 1e7, 'In Millions': 1e6
        }
        detected = st.session_state.get('detected_unit')
        default_idx = 0
        if detected:
            for i, k in enumerate(unit_opts.keys()):
                if detected.lower() in k.lower():
                    default_idx = i
                    break
        unit_choice = st.sidebar.selectbox(
            "Data Unit", list(unit_opts.keys()), index=default_idx
        )
        st.session_state.unit_scale = unit_opts[unit_choice]
        Cfg.UNIT_SCALE = unit_opts[unit_choice]

        Cfg.DEBUG = st.sidebar.checkbox("Debug Mode", Cfg.DEBUG)

        # Clear data
        if st.sidebar.button("🗑️ Clear All Data"):
            for k in [
                'data', 'mappings', 'pn_mappings', 'pn_results',
                'forecast_results', 'validation_report'
            ]:
                st.session_state[k] = None
            st.rerun()

        # Performance stats
        if Cfg.DEBUG:
            with st.sidebar.expander("📊 Performance"):
                perf = Perf.summary()
                for name, stats in perf.items():
                    st.text(
                        f"{name}: {stats['avg']:.3f}s avg "
                        f"({stats['count']} calls)"
                    )

    def _process_files(self, files):
        

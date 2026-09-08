import hashlib
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / 'data'
DEFAULT_START_YEAR = 2015
DEFAULT_END_YEAR = pd.Timestamp.now(tz='America/Los_Angeles').year
DEFAULT_MAX_FRESHNESS_DAYS = 365


class BaseCleaner(ABC):
    def __init__(
        self,
        source_path,
        start_year=DEFAULT_START_YEAR,
        end_year=DEFAULT_END_YEAR,
        max_freshness_days=DEFAULT_MAX_FRESHNESS_DAYS
    ):
        if start_year > end_year:
            raise ValueError('start_year must be less than or equal to end_year')

        self.source_path = Path(source_path)
        self.start_year = start_year
        self.end_year = end_year
        self.max_freshness_days = max_freshness_days
        self.source_stat = None

    @abstractmethod
    def clean(self, *args, **kwargs):
        pass

    @staticmethod
    def validate_columns(df, required_columns, dataset_name):
        missing_columns = [
            col for col in required_columns if col not in df.columns
        ]

        if missing_columns:
            raise ValueError(
                f'{dataset_name} is missing required columns: {missing_columns}'
            )

    @staticmethod
    def parse_datetime(
        column,
        column_name,
        required=False,
        date_format='mixed'
    ):
        parsed = pd.to_datetime(column, format=date_format, errors='coerce')
        invalid = column.notna() & parsed.isna()

        if invalid.any():
            examples = column[invalid].astype(str).head(3).tolist()
            raise ValueError(
                f'{column_name} contains {invalid.sum()} invalid values. '
                f'Examples: {examples}'
            )

        if required and parsed.isna().any():
            raise ValueError(
                f'{column_name} contains {parsed.isna().sum()} missing values'
            )

        return parsed

    @staticmethod
    def validate_freshness(dates, dataset_name, max_freshness_days):
        latest_date = dates.max()

        if latest_date.tzinfo is not None:
            latest_date = latest_date.tz_convert('America/Los_Angeles')

        latest_date = latest_date.date()
        current_date = pd.Timestamp.now(tz='America/Los_Angeles').date()
        freshness_days = (current_date - latest_date).days

        if freshness_days < 0:
            raise ValueError(f'{dataset_name} contains dates in the future')

        if freshness_days > max_freshness_days:
            raise ValueError(
                f'{dataset_name} is {freshness_days} days stale; '
                f'the limit is {max_freshness_days} days'
            )

        return freshness_days

    def filter_year_window(self, df, year_column='YEAR'):
        excluded_before_window = int(
            (df[year_column] < self.start_year).sum()
        )
        excluded_after_window = int(
            (df[year_column] > self.end_year).sum()
        )
        filtered = df[
            (df[year_column] >= self.start_year)
            & (df[year_column] <= self.end_year)
        ].copy()

        return filtered, excluded_before_window, excluded_after_window

    def capture_source_stat(self):
        self.source_stat = self.source_path.stat()

    def validate_source_unchanged(self, action):
        if self.source_stat is None:
            raise RuntimeError('Source metadata was not captured before reading')

        current_source_stat = self.source_path.stat()
        if (
            self.source_stat.st_size != current_source_stat.st_size
            or self.source_stat.st_mtime_ns != current_source_stat.st_mtime_ns
        ):
            raise ValueError(f'{self.source_path.name} changed {action}')

    def source_details(
        self,
        row_count,
        min_date,
        max_date,
        freshness_days
    ):
        self.validate_source_unchanged('after it was read')
        source_stat_before = self.source_path.stat()

        digest = hashlib.sha256()
        with self.source_path.open('rb') as source_file:
            for chunk in iter(lambda: source_file.read(1024 * 1024), b''):
                digest.update(chunk)

        source_stat = self.source_path.stat()
        if (
            source_stat_before.st_size != source_stat.st_size
            or source_stat_before.st_mtime_ns != source_stat.st_mtime_ns
        ):
            raise ValueError(
                f'{self.source_path.name} changed while its checksum was calculated'
            )

        try:
            source_name = str(self.source_path.relative_to(BASE_DIR))
        except ValueError:
            source_name = str(self.source_path)

        return {
            'path': source_name,
            'bytes': source_stat.st_size,
            'modified_utc': datetime.fromtimestamp(
                source_stat.st_mtime,
                tz=timezone.utc
            ).isoformat(),
            'sha256': digest.hexdigest(),
            'rows': row_count,
            'min_incident_date': str(min_date.date()),
            'max_incident_date': str(max_date.date()),
            'freshness_days': freshness_days
        }

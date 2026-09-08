import pandas as pd

from cleaning_base import (
    DATA_DIR,
    DEFAULT_END_YEAR,
    DEFAULT_MAX_FRESHNESS_DAYS,
    DEFAULT_START_YEAR,
    BaseCleaner
)


VEHICLE_SOURCE = DATA_DIR / 'raw/SDOT_Vehicle.csv'
VEHICLE_TYPE_MAP = {
    'Passenger Car': 'Passenger Vehicle',
    'Taxi': 'Passenger Vehicle',
    'Pickup, Panel Truck or Vannette Under 10,000 lbs': 'Light Truck/Van',
    'Motorcycle': 'Two-Wheeled',
    'Moped': 'Two-Wheeled',
    'Scooter Bike': 'Two-Wheeled',
    'Truck (Flatbed, Van, etc)': 'Commercial Truck',
    'Truck - Double trailer Combinations': 'Commercial Truck',
    'Truck Tractor': 'Commercial Truck',
    'Truck Tractor and Semi-Trailer': 'Commercial Truck',
    'Truck and Trailer': 'Commercial Truck',
    'Bus or Motor Stage': 'Bus',
    'School Bus': 'Bus',
    'Farm Tractor and/or Farm Equipment': 'Farm Equipment',
    'Other': 'Other',
    'Not Stated': 'Unknown',
    'Railway Vehicle': 'Railway Vehicle'
}


class VehicleCleaner(BaseCleaner):
    def __init__(
        self,
        start_year=DEFAULT_START_YEAR,
        end_year=DEFAULT_END_YEAR,
        max_freshness_days=DEFAULT_MAX_FRESHNESS_DAYS,
        source_path=VEHICLE_SOURCE
    ):
        super().__init__(
            source_path,
            start_year,
            end_year,
            max_freshness_days
        )

    def clean(self, collision_keys):
        self.capture_source_stat()
        required_columns = [
            'COLLISIONVEHDETKEY',
            'COLDETKEY',
            'ST_VEH_TYPE_DESC',
            'Incident Date'
        ]
        df_vehicle = pd.read_csv(
            self.source_path,
            usecols=required_columns,
            dtype={
                'COLLISIONVEHDETKEY': 'Int64',
                'COLDETKEY': 'Int64',
                'ST_VEH_TYPE_DESC': 'string'
            }
        )
        self.validate_source_unchanged('while it was being read')
        self.validate_columns(df_vehicle, required_columns, 'Raw vehicle data')

        source_rows = len(df_vehicle)
        df_vehicle['Incident Date'] = self.parse_datetime(
            df_vehicle['Incident Date'],
            'Incident Date',
            required=True,
            date_format='%m/%d/%Y %I:%M:%S %p'
        )
        freshness_days = self.validate_freshness(
            df_vehicle['Incident Date'],
            'Raw vehicle data',
            self.max_freshness_days
        )
        min_incident_date = df_vehicle['Incident Date'].min()
        max_incident_date = df_vehicle['Incident Date'].max()
        df_vehicle['YEAR'] = df_vehicle['Incident Date'].dt.year
        df_vehicle, excluded_before_window, excluded_after_window = (
            self.filter_year_window(df_vehicle)
        )

        df_vehicle['ST_VEH_TYPE_DESC_RAW'] = df_vehicle['ST_VEH_TYPE_DESC']
        vehicle_type = df_vehicle['ST_VEH_TYPE_DESC'].str.strip()
        unmapped_vehicle_types = (
            vehicle_type.notna()
            & ~vehicle_type.isin(VEHICLE_TYPE_MAP)
        )
        if unmapped_vehicle_types.any():
            values = (
                vehicle_type[unmapped_vehicle_types]
                .drop_duplicates()
                .tolist()
            )
            raise ValueError(
                f'Raw vehicle data has unmapped vehicle types: {values}'
            )

        df_vehicle['VEHICLE_CATEGORY'] = (
            vehicle_type.map(VEHICLE_TYPE_MAP).fillna('Unknown')
        )
        df_vehicle['ST_VEH_TYPE_DESC'] = df_vehicle['VEHICLE_CATEGORY']
        df_vehicle = df_vehicle[[
            'COLLISIONVEHDETKEY',
            'COLDETKEY',
            'ST_VEH_TYPE_DESC_RAW',
            'ST_VEH_TYPE_DESC',
            'VEHICLE_CATEGORY',
            'Incident Date',
            'YEAR'
        ]]
        orphan_rows = ~df_vehicle['COLDETKEY'].isin(collision_keys)
        orphan_vehicle_df = df_vehicle[orphan_rows].copy()
        df_vehicle = df_vehicle[~orphan_rows].copy()
        self.validate_vehicle_data(df_vehicle)

        stats = {
            'source': self.source_details(
                source_rows,
                min_incident_date,
                max_incident_date,
                freshness_days
            ),
            'cleaned_rows': len(df_vehicle),
            'canonical_format': 'parquet',
            'unique_collision_keys': int(df_vehicle['COLDETKEY'].nunique()),
            'excluded_before_window': excluded_before_window,
            'excluded_after_window': excluded_after_window,
            'unknown_vehicle_type_rows': int(
                (df_vehicle['VEHICLE_CATEGORY'] == 'Unknown').sum()
            ),
            'orphan_vehicle_rows': len(orphan_vehicle_df),
            'orphan_collision_keys': int(
                orphan_vehicle_df['COLDETKEY'].nunique()
            ),
            'vehicle_categories': {
                str(key): int(value)
                for key, value in df_vehicle['VEHICLE_CATEGORY']
                .value_counts()
                .items()
            }
        }

        return df_vehicle, orphan_vehicle_df, stats

    @classmethod
    def validate_vehicle_data(cls, df_vehicle):
        required_columns = [
            'COLLISIONVEHDETKEY',
            'COLDETKEY',
            'ST_VEH_TYPE_DESC_RAW',
            'ST_VEH_TYPE_DESC',
            'VEHICLE_CATEGORY',
            'Incident Date',
            'YEAR'
        ]
        cls.validate_columns(
            df_vehicle,
            required_columns,
            'Cleaned vehicle data'
        )

        if df_vehicle.empty:
            raise ValueError('Cleaned vehicle data contains no rows')

        required_values = [
            'COLLISIONVEHDETKEY',
            'COLDETKEY',
            'VEHICLE_CATEGORY',
            'YEAR'
        ]
        if df_vehicle[required_values].isna().any().any():
            raise ValueError(
                'Cleaned vehicle data contains missing required values'
            )

        if df_vehicle['COLLISIONVEHDETKEY'].duplicated().any():
            raise ValueError(
                'Cleaned vehicle data contains duplicate vehicle keys'
            )

        valid_categories = set(VEHICLE_TYPE_MAP.values()) | {'Unknown'}
        if not df_vehicle['VEHICLE_CATEGORY'].isin(valid_categories).all():
            raise ValueError(
                'Cleaned vehicle data contains unexpected vehicle categories'
            )

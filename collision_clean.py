import geopandas as gpd
import pandas as pd

from cleaning_base import (
    DATA_DIR,
    DEFAULT_END_YEAR,
    DEFAULT_MAX_FRESHNESS_DAYS,
    DEFAULT_START_YEAR,
    BaseCleaner
)


SEATTLE_BOUNDS = {
    'x_min': -122.45,
    'x_max': -122.20,
    'y_min': 47.45,
    'y_max': 47.75
}


def get_collision_source():
    source_paths = [
        DATA_DIR / 'raw/SDOT_Collisions_All_Years.geojson',
        DATA_DIR / 'raw/SDOT_Collision_All_Years.geojson'
    ]

    for path in source_paths:
        if path.exists():
            return path

    raise FileNotFoundError(
        'Could not find SDOT_Collisions_All_Years.geojson in data/raw'
    )


class CollisionCleaner(BaseCleaner):
    def __init__(
        self,
        start_year=DEFAULT_START_YEAR,
        end_year=DEFAULT_END_YEAR,
        max_freshness_days=DEFAULT_MAX_FRESHNESS_DAYS,
        source_path=None
    ):
        source_path = source_path or get_collision_source()
        super().__init__(
            source_path,
            start_year,
            end_year,
            max_freshness_days
        )

    def clean(self):
        self.capture_source_stat()
        gdf = gpd.read_file(self.source_path)
        self.validate_source_unchanged('while it was being read')

        required_columns = [
            'COLDETKEY',
            'LOCATION',
            'MAXSEVERITYCODE',
            'MAXSEVERITYDESC',
            'PERSONCOUNT',
            'PEDCOUNT',
            'PEDCYLCOUNT',
            'VEHCOUNT',
            'INJURIES',
            'SERIOUSINJURIES',
            'FATALITIES',
            'INCDATE',
            'INCDTTM',
            'JUNCTIONTYPE',
            'SDOT_COLCODE',
            'SDOT_COLDESC',
            'UNDERINFL',
            'WEATHER',
            'ROADCOND',
            'LIGHTCOND',
            'geometry'
        ]
        self.validate_columns(gdf, required_columns, 'Raw collision data')

        source_rows = len(gdf)
        gdf['INCDATE'] = self.parse_datetime(
            gdf['INCDATE'],
            'INCDATE',
            required=True
        )
        gdf['INCDTTM'] = self.parse_datetime(gdf['INCDTTM'], 'INCDTTM')
        freshness_days = self.validate_freshness(
            gdf['INCDATE'],
            'Raw collision data',
            self.max_freshness_days
        )

        min_incident_date = gdf['INCDATE'].min()
        max_incident_date = gdf['INCDATE'].max()
        gdf['YEAR'] = gdf['INCDATE'].dt.year
        gdf['MONTH'] = gdf['INCDATE'].dt.month
        gdf['DAY'] = gdf['INCDATE'].dt.day
        gdf['HOUR'] = gdf['INCDTTM'].dt.hour
        gdf, excluded_before_window, excluded_after_window = (
            self.filter_year_window(gdf)
        )

        moderate_nan_cols = ['WEATHER', 'ROADCOND', 'LIGHTCOND', 'JUNCTIONTYPE']
        for col in moderate_nan_cols:
            gdf[f'{col}_RAW'] = gdf[col]
            gdf[col] = gdf[col].astype('string').str.strip()
            gdf[col] = gdf[col].mask(gdf[col] == '').fillna('Unknown')

        gdf['UNDERINFL_RAW'] = gdf['UNDERINFL']
        under_influence = gdf['UNDERINFL'].astype('string').str.strip().str.upper()
        under_influence_map = {
            'N': 'No',
            '0': 'No',
            'Y': 'Yes',
            '1': 'Yes',
            'UNKNOWN': 'Unknown'
        }
        unmapped_under_influence = (
            under_influence.notna()
            & ~under_influence.isin(under_influence_map)
        )
        if unmapped_under_influence.any():
            values = (
                under_influence[unmapped_under_influence]
                .drop_duplicates()
                .tolist()
            )
            raise ValueError(
                f'Raw collision data has unmapped UNDERINFL values: {values}'
            )

        gdf['UNDERINFL'] = (
            under_influence.map(under_influence_map).fillna('Unknown')
        )
        gdf['Day_of_Week'] = gdf['INCDATE'].dt.day_name()
        gdf['Season'] = gdf['MONTH'].map(self.get_season)
        gdf = self.add_geometry_quality(gdf)
        self.validate_collision_data(gdf, 'Cleaned collision data')

        spatial_excluded = gdf[~gdf['HAS_VALID_GEOMETRY']]
        midnight = (
            gdf['INCDTTM'].dt.hour.eq(0)
            & gdf['INCDTTM'].dt.minute.eq(0)
        )
        stats = {
            'source': self.source_details(
                source_rows,
                min_incident_date,
                max_incident_date,
                freshness_days
            ),
            'cleaned_rows': len(gdf),
            'canonical_format': 'parquet',
            'max_cleaned_incident_date': str(gdf['INCDATE'].max().date()),
            'excluded_before_window': excluded_before_window,
            'excluded_after_window': excluded_after_window,
            'severity_missing_rows': int(gdf['MAXSEVERITYCODE'].isna().sum()),
            'midnight_time_rows': int(midnight.sum()),
            'geometry_status': {
                str(key): int(value)
                for key, value in gdf['GEOMETRY_STATUS'].value_counts().items()
            },
            'spatial_exclusions_by_year': {
                str(key): int(value)
                for key, value in spatial_excluded.groupby('YEAR').size().items()
            },
            'spatial_exclusions_by_severity': {
                str(key): int(value)
                for key, value in spatial_excluded['MAXSEVERITYDESC']
                .fillna('Missing')
                .value_counts()
                .items()
            },
            'business_rule_warnings': {
                'fatalities_greater_than_person_count': int(
                    (gdf['FATALITIES'] > gdf['PERSONCOUNT']).sum()
                ),
                'pedestrian_and_cyclist_count_greater_than_person_count': int(
                    (
                        (gdf['PEDCOUNT'] + gdf['PEDCYLCOUNT'])
                        > gdf['PERSONCOUNT']
                    ).sum()
                ),
                'serious_injuries_greater_than_injuries': int(
                    (gdf['SERIOUSINJURIES'] > gdf['INJURIES']).sum()
                )
            }
        }

        return gdf, stats

    @staticmethod
    def get_season(month):
        if month in [12, 1, 2]:
            return 'Winter'
        if month in [3, 4, 5]:
            return 'Spring'
        if month in [6, 7, 8]:
            return 'Summer'
        if month in [9, 10, 11]:
            return 'Fall'

    @staticmethod
    def add_geometry_quality(gdf):
        if gdf.crs is None:
            raise ValueError('Collision data is missing its CRS')

        if gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs(epsg=4326)

        geometry_status = pd.Series('valid', index=gdf.index, dtype='object')
        missing = gdf.geometry.isna()
        empty = ~missing & gdf.geometry.is_empty
        invalid_type = ~missing & ~empty & (gdf.geometry.geom_type != 'Point')
        invalid = ~missing & ~empty & ~gdf.geometry.is_valid
        valid_point = ~missing & ~empty & ~invalid_type & ~invalid

        outside_seattle = pd.Series(False, index=gdf.index)
        points = gdf.loc[valid_point].geometry
        outside_seattle.loc[valid_point] = (
            (points.x < SEATTLE_BOUNDS['x_min'])
            | (points.x > SEATTLE_BOUNDS['x_max'])
            | (points.y < SEATTLE_BOUNDS['y_min'])
            | (points.y > SEATTLE_BOUNDS['y_max'])
        )

        geometry_status.loc[missing] = 'missing'
        geometry_status.loc[empty] = 'empty'
        geometry_status.loc[invalid_type] = 'invalid_type'
        geometry_status.loc[invalid] = 'invalid'
        geometry_status.loc[outside_seattle] = 'outside_seattle'

        gdf['GEOMETRY_STATUS'] = geometry_status
        gdf['HAS_VALID_GEOMETRY'] = geometry_status == 'valid'

        return gdf

    @classmethod
    def validate_collision_data(cls, gdf, dataset_name):
        required_columns = [
            'COLDETKEY',
            'MAXSEVERITYCODE',
            'MAXSEVERITYDESC',
            'INCDATE',
            'INCDTTM',
            'YEAR',
            'MONTH',
            'DAY',
            'HOUR',
            'INJURIES',
            'SERIOUSINJURIES',
            'FATALITIES',
            'PEDCOUNT',
            'PEDCYLCOUNT',
            'UNDERINFL',
            'GEOMETRY_STATUS',
            'HAS_VALID_GEOMETRY',
            'geometry'
        ]
        cls.validate_columns(gdf, required_columns, dataset_name)

        if gdf.empty:
            raise ValueError(f'{dataset_name} contains no rows')

        if gdf['COLDETKEY'].isna().any():
            raise ValueError(f'{dataset_name} contains missing COLDETKEY values')

        if gdf['COLDETKEY'].duplicated().any():
            raise ValueError(f'{dataset_name} contains duplicate COLDETKEY values')

        required_value_columns = [
            'COLDETKEY',
            'INCDATE',
            'YEAR',
            'MONTH',
            'DAY',
            'INJURIES',
            'SERIOUSINJURIES',
            'FATALITIES',
            'PEDCOUNT',
            'PEDCYLCOUNT'
        ]
        missing_values = gdf[required_value_columns].isna().sum()
        missing_values = missing_values[missing_values > 0].to_dict()

        if missing_values:
            raise ValueError(
                f'{dataset_name} contains missing required values: {missing_values}'
            )

        count_columns = [
            'INJURIES',
            'SERIOUSINJURIES',
            'FATALITIES',
            'PEDCOUNT',
            'PEDCYLCOUNT'
        ]
        if (gdf[count_columns] < 0).any().any():
            raise ValueError(f'{dataset_name} contains negative count values')

        for col in ['MAXSEVERITYCODE', 'MAXSEVERITYDESC']:
            missing_rate = gdf[col].isna().mean()
            if missing_rate > 0.01:
                raise ValueError(
                    f'{dataset_name} has {missing_rate:.1%} missing values in {col}'
                )

        if not gdf['UNDERINFL'].isin(['Yes', 'No', 'Unknown']).all():
            raise ValueError(
                f'{dataset_name} contains unexpected UNDERINFL values'
            )

        valid_geometry_status = [
            'valid',
            'missing',
            'empty',
            'invalid_type',
            'invalid',
            'outside_seattle'
        ]
        if not gdf['GEOMETRY_STATUS'].isin(valid_geometry_status).all():
            raise ValueError(
                f'{dataset_name} contains unexpected geometry status values'
            )

        hours = gdf['HOUR'].dropna()
        if len(gdf) >= 100 and hours.nunique() <= 1:
            raise ValueError(
                f'{dataset_name} has only '
                f'{hours.nunique()} distinct non-null HOUR values'
            )

        if not hours.between(0, 23).all():
            raise ValueError(f'{dataset_name} contains HOUR values outside 0-23')

    @classmethod
    def process(cls, gdf):
        gdf = gdf[gdf['HAS_VALID_GEOMETRY']].copy()
        gdf['X'] = gdf.geometry.x
        gdf['Y'] = gdf.geometry.y
        gdf['SEVERITY'] = (
            gdf['INJURIES']
            + (gdf['SERIOUSINJURIES'] * 3)
            + (gdf['FATALITIES'] * 5)
        )
        gdf['TOTAL_PED'] = gdf['PEDCOUNT'] + gdf['PEDCYLCOUNT']
        gdf['lat_bin'] = pd.cut(gdf['Y'], bins=150).astype(str)
        gdf['lon_bin'] = pd.cut(gdf['X'], bins=150).astype(str)

        cls.validate_collision_data(gdf, 'Processed collision data')
        processed_columns = [
            'X',
            'Y',
            'SEVERITY',
            'TOTAL_PED',
            'lat_bin',
            'lon_bin'
        ]
        cls.validate_columns(gdf, processed_columns, 'Processed collision data')

        if gdf[processed_columns].isna().any().any():
            raise ValueError(
                'Processed collision data contains missing derived values'
            )

        if gdf['INCDTTM'].isna().mean() > 0.50:
            raise ValueError(
                'Processed collision data has more than 50% missing INCDTTM values'
            )

        output_columns = [
            'INCKEY',
            'COLDETKEY',
            'REPORTNO',
            'LOCATION',
            'MAXSEVERITYCODE',
            'MAXSEVERITYDESC',
            'PERSONCOUNT',
            'PEDCOUNT',
            'PEDCYLCOUNT',
            'VEHCOUNT',
            'INJURIES',
            'SERIOUSINJURIES',
            'FATALITIES',
            'INCDATE',
            'INCDTTM',
            'JUNCTIONTYPE',
            'ST_COLLISIONTYPE',
            'SDOT_COLCODE',
            'SDOT_COLDESC',
            'INATTENTIONIND',
            'UNDERINFL',
            'WEATHER',
            'ROADCOND',
            'LIGHTCOND',
            'PEDROWNOTGRNT',
            'SPEEDING',
            'HITPARKEDCAR',
            'SHAREDMICROMOBILITYDESC',
            'YEAR',
            'MONTH',
            'DAY',
            'HOUR',
            'Day_of_Week',
            'Season',
            'GEOMETRY_STATUS',
            'HAS_VALID_GEOMETRY',
            'X',
            'Y',
            'SEVERITY',
            'TOTAL_PED',
            'lat_bin',
            'lon_bin',
            'geometry'
        ]
        output_columns = [col for col in output_columns if col in gdf.columns]

        return gdf[output_columns].copy()

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from cleaning_base import (
    DATA_DIR,
    DEFAULT_END_YEAR,
    DEFAULT_MAX_FRESHNESS_DAYS,
    DEFAULT_START_YEAR
)
from collision_clean import CollisionCleaner, get_collision_source
from vehicle_clean import VEHICLE_SOURCE, VehicleCleaner


__all__ = [
    'All_Years_Clean',
    'Processed_Clean',
    'Vehicles_Clean',
    'get_collision_source',
    'main'
]


def All_Years_Clean(
    start_year=DEFAULT_START_YEAR,
    end_year=DEFAULT_END_YEAR,
    max_freshness_days=DEFAULT_MAX_FRESHNESS_DAYS,
    source_path=None
):
    cleaner = CollisionCleaner(
        start_year,
        end_year,
        max_freshness_days,
        source_path
    )
    return cleaner.clean()


def Vehicles_Clean(
    collision_keys,
    start_year=DEFAULT_START_YEAR,
    end_year=DEFAULT_END_YEAR,
    max_freshness_days=DEFAULT_MAX_FRESHNESS_DAYS,
    source_path=VEHICLE_SOURCE
):
    cleaner = VehicleCleaner(
        start_year,
        end_year,
        max_freshness_days,
        source_path
    )
    return cleaner.clean(collision_keys)


def Processed_Clean(gdf):
    return CollisionCleaner.process(gdf)


def write_dataframe(df, path, output_format):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(
        f'.{path.stem}.{uuid4().hex}.tmp{path.suffix}'
    )

    try:
        if output_format == 'parquet':
            df.to_parquet(temp_path, index=False)
        elif output_format == 'csv':
            df.to_csv(temp_path, index=False)
        elif output_format == 'geojson':
            df.to_file(temp_path, driver='GeoJSON', index=False)
        else:
            raise ValueError(f'Unsupported output format: {output_format}')

        temp_path.replace(path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def write_json(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(
        f'.{path.stem}.{uuid4().hex}.tmp{path.suffix}'
    )

    try:
        temp_path.write_text(json.dumps(data, indent=2) + '\n')
        temp_path.replace(path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def write_outputs(
    collision_gdf,
    vehicle_df,
    orphan_vehicle_df,
    processed_gdf,
    manifest,
    output_root=DATA_DIR,
    write_geojson=True
):
    output_root = Path(output_root)
    cleaned_dir = output_root / 'cleaned'
    processed_dir = output_root / 'processed'
    spatial_excluded = collision_gdf[~collision_gdf['HAS_VALID_GEOMETRY']]

    output_paths = {
        'collision_cleaned_parquet': cleaned_dir / 'Collision_All_Filtered.parquet',
        'collision_spatial_excluded': (
            cleaned_dir / 'Collision_Spatial_Excluded.parquet'
        ),
        'vehicle_cleaned_parquet': cleaned_dir / 'Vehicle_Filtered.parquet',
        'vehicle_cleaned_csv': cleaned_dir / 'Vehicle_Filtered.csv',
        'vehicle_orphaned': cleaned_dir / 'Vehicle_Orphaned.parquet',
        'collision_processed_parquet': (
            processed_dir / 'Collision_Processed.parquet'
        )
    }

    write_dataframe(
        collision_gdf,
        output_paths['collision_cleaned_parquet'],
        'parquet'
    )
    write_dataframe(
        spatial_excluded,
        output_paths['collision_spatial_excluded'],
        'parquet'
    )
    write_dataframe(
        vehicle_df,
        output_paths['vehicle_cleaned_parquet'],
        'parquet'
    )
    write_dataframe(vehicle_df, output_paths['vehicle_cleaned_csv'], 'csv')
    write_dataframe(
        orphan_vehicle_df,
        output_paths['vehicle_orphaned'],
        'parquet'
    )
    write_dataframe(
        processed_gdf,
        output_paths['collision_processed_parquet'],
        'parquet'
    )

    if write_geojson:
        output_paths['collision_cleaned_geojson'] = (
            cleaned_dir / 'Collision_All_Filtered.geojson'
        )
        output_paths['collision_processed_geojson'] = (
            processed_dir / 'Collision_Processed.geojson'
        )
        write_dataframe(
            collision_gdf,
            output_paths['collision_cleaned_geojson'],
            'geojson'
        )
        write_dataframe(
            processed_gdf,
            output_paths['collision_processed_geojson'],
            'geojson'
        )

    manifest['outputs'] = {
        name: {
            'path': str(path),
            'bytes': path.stat().st_size
        }
        for name, path in output_paths.items()
    }
    manifest_path = processed_dir / 'cleaning_manifest.json'
    write_json(manifest, manifest_path)

    return manifest_path


def main(
    start_year=DEFAULT_START_YEAR,
    end_year=DEFAULT_END_YEAR,
    max_freshness_days=DEFAULT_MAX_FRESHNESS_DAYS,
    write_geojson=True,
    output_root=DATA_DIR,
    collision_source=None,
    vehicle_source=VEHICLE_SOURCE
):
    if start_year > end_year:
        raise ValueError('start_year must be less than or equal to end_year')

    collision_cleaner = CollisionCleaner(
        start_year,
        end_year,
        max_freshness_days,
        collision_source
    )
    collision_gdf, collision_stats = collision_cleaner.clean()

    vehicle_cleaner = VehicleCleaner(
        start_year,
        end_year,
        max_freshness_days,
        vehicle_source
    )
    vehicle_df, orphan_vehicle_df, vehicle_stats = vehicle_cleaner.clean(
        collision_gdf['COLDETKEY']
    )
    processed_gdf = collision_cleaner.process(collision_gdf)

    manifest = {
        'run_timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'analysis_window': {
            'start_year': start_year,
            'end_year': end_year,
            'max_freshness_days': max_freshness_days,
            'latest_year_is_partial': (
                collision_stats['max_cleaned_incident_date']
                < f'{end_year}-12-31'
            )
        },
        'collision_data': collision_stats,
        'vehicle_data': vehicle_stats,
        'processed_data': {
            'rows': len(processed_gdf),
            'columns': list(processed_gdf.columns),
            'canonical_format': 'parquet'
        }
    }
    manifest_path = write_outputs(
        collision_gdf,
        vehicle_df,
        orphan_vehicle_df,
        processed_gdf,
        manifest,
        output_root,
        write_geojson
    )

    print(f'Cleaned {len(collision_gdf):,} collision rows')
    print(f'Cleaned {len(vehicle_df):,} vehicle rows')
    print(f'Published {len(processed_gdf):,} spatial collision rows')
    print(f'Quality manifest: {manifest_path}')

    return manifest


def parse_args():
    parser = argparse.ArgumentParser(description='Clean SDOT collision data')
    parser.add_argument('--start-year', type=int, default=DEFAULT_START_YEAR)
    parser.add_argument('--end-year', type=int, default=DEFAULT_END_YEAR)
    parser.add_argument(
        '--max-freshness-days',
        type=int,
        default=DEFAULT_MAX_FRESHNESS_DAYS
    )
    parser.add_argument(
        '--skip-geojson',
        action='store_true',
        help='Only write the canonical Parquet outputs'
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(
        start_year=args.start_year,
        end_year=args.end_year,
        max_freshness_days=args.max_freshness_days,
        write_geojson=not args.skip_geojson
    )

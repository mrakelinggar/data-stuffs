"""
Data Validator for NYC Citi Bike Trip Data

Validates CSV schemas, identifies missing values, detects outliers,
and generates validation reports.
"""

import zipfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import io

import pandas as pd
from loguru import logger


class BikeDataValidator:
    """Validator for Citi Bike trip data files."""
    
    # New schema (2021+)
    NEW_SCHEMA = [
        "ride_id",
        "rideable_type", 
        "started_at",
        "ended_at",
        "start_station_name",
        "start_station_id",
        "end_station_name",
        "end_station_id",
        "start_lat",
        "start_lng",
        "end_lat",
        "end_lng",
        "member_casual"
    ]
    
    # Old schema (pre-2021)
    OLD_SCHEMA = [
        "tripduration",
        "starttime",
        "stoptime",
        "start station id",
        "start station name",
        "start station latitude",
        "start station longitude",
        "end station id",
        "end station name",
        "end station latitude",
        "end station longitude",
        "bikeid",
        "usertype",
        "birth year",
        "gender"
    ]
    
    # NYC bounding box (approximate)
    NYC_BOUNDS = {
        "lat_min": 40.4774,
        "lat_max": 40.9176,
        "lng_min": -74.2591,
        "lng_max": -73.7004
    }
    
    def __init__(self, data_dir: Path = Path("data/raw")):
        """
        Initialize validator.
        
        Args:
            data_dir: Directory containing downloaded zip files
        """
        self.data_dir = Path(data_dir)
        self.validation_dir = Path("data/validation")
        self.validation_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initialized BikeDataValidator with data_dir: {self.data_dir}")
    
    def detect_schema(self, df: pd.DataFrame) -> str:
        """
        Detect which schema the dataframe uses.
        
        Args:
            df: DataFrame to check
            
        Returns:
            "new" or "old" schema type
        """
        columns_lower = [col.lower() for col in df.columns]
        
        # Check for new schema indicators
        if "ride_id" in columns_lower or "rideable_type" in columns_lower:
            return "new"
        
        # Check for old schema indicators
        if "tripduration" in columns_lower or "bikeid" in columns_lower:
            return "old"
        
        # Default to new if unclear
        logger.warning("Could not definitively detect schema, assuming new format")
        return "new"
    
    def validate_schema(self, df: pd.DataFrame, filename: str) -> Dict:
        """
        Validate DataFrame schema against expected columns.
        
        Args:
            df: DataFrame to validate
            filename: Name of file being validated
            
        Returns:
            Dict with validation results
        """
        schema_type = self.detect_schema(df)
        expected_cols = self.NEW_SCHEMA if schema_type == "new" else self.OLD_SCHEMA
        
        # Case-insensitive column matching
        df_cols_lower = [col.lower() for col in df.columns]
        expected_cols_lower = [col.lower() for col in expected_cols]
        
        missing_cols = set(expected_cols_lower) - set(df_cols_lower)
        extra_cols = set(df_cols_lower) - set(expected_cols_lower)
        
        return {
            "filename": filename,
            "schema_type": schema_type,
            "total_columns": len(df.columns),
            "expected_columns": len(expected_cols),
            "missing_columns": list(missing_cols),
            "extra_columns": list(extra_cols),
            "valid": len(missing_cols) == 0
        }
    
    def check_missing_values(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate percentage of missing values per column.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dict mapping column name to % missing
        """
        missing_pct = (df.isnull().sum() / len(df) * 100).round(2)
        return missing_pct.to_dict()
    
    def detect_outliers(self, df: pd.DataFrame, schema_type: str) -> Dict:
        """
        Detect outliers in trip data.
        
        Args:
            df: DataFrame to analyze
            schema_type: "new" or "old"
            
        Returns:
            Dict with outlier counts and percentages
        """
        outliers = {}
        total_rows = len(df)
        
        if schema_type == "new":
            # Calculate trip duration from timestamps
            if "started_at" in df.columns and "ended_at" in df.columns:
                df["started_at"] = pd.to_datetime(df["started_at"], errors="coerce")
                df["ended_at"] = pd.to_datetime(df["ended_at"], errors="coerce")
                duration_sec = (df["ended_at"] - df["started_at"]).dt.total_seconds()
                
                # Trip duration outliers (< 1 min or > 24 hours)
                short_trips = (duration_sec < 60).sum()
                long_trips = (duration_sec > 86400).sum()
                negative_trips = (duration_sec < 0).sum()
                
                outliers["short_trips"] = {
                    "count": int(short_trips),
                    "percent": round(short_trips / total_rows * 100, 2)
                }
                outliers["long_trips"] = {
                    "count": int(long_trips),
                    "percent": round(long_trips / total_rows * 100, 2)
                }
                outliers["negative_duration"] = {
                    "count": int(negative_trips),
                    "percent": round(negative_trips / total_rows * 100, 2)
                }
            
            # Coordinate outliers (outside NYC bounds)
            coord_outliers = 0
            for lat_col, lng_col in [("start_lat", "start_lng"), ("end_lat", "end_lng")]:
                if lat_col in df.columns and lng_col in df.columns:
                    out_of_bounds = (
                        (df[lat_col] < self.NYC_BOUNDS["lat_min"]) |
                        (df[lat_col] > self.NYC_BOUNDS["lat_max"]) |
                        (df[lng_col] < self.NYC_BOUNDS["lng_min"]) |
                        (df[lng_col] > self.NYC_BOUNDS["lng_max"])
                    ).sum()
                    coord_outliers += out_of_bounds
            
            outliers["coordinate_outliers"] = {
                "count": int(coord_outliers),
                "percent": round(coord_outliers / (total_rows * 2) * 100, 2)  # 2 coords per trip
            }
        
        else:  # old schema
            # Trip duration outliers
            if "tripduration" in df.columns:
                short_trips = (df["tripduration"] < 60).sum()
                long_trips = (df["tripduration"] > 86400).sum()
                
                outliers["short_trips"] = {
                    "count": int(short_trips),
                    "percent": round(short_trips / total_rows * 100, 2)
                }
                outliers["long_trips"] = {
                    "count": int(long_trips),
                    "percent": round(long_trips / total_rows * 100, 2)
                }
            
            # Coordinate outliers
            coord_outliers = 0
            for lat_col, lng_col in [
                ("start station latitude", "start station longitude"),
                ("end station latitude", "end station longitude")
            ]:
                if lat_col in df.columns and lng_col in df.columns:
                    out_of_bounds = (
                        (df[lat_col] < self.NYC_BOUNDS["lat_min"]) |
                        (df[lat_col] > self.NYC_BOUNDS["lat_max"]) |
                        (df[lng_col] < self.NYC_BOUNDS["lng_min"]) |
                        (df[lng_col] > self.NYC_BOUNDS["lng_max"])
                    ).sum()
                    coord_outliers += out_of_bounds
            
            outliers["coordinate_outliers"] = {
                "count": int(coord_outliers),
                "percent": round(coord_outliers / (total_rows * 2) * 100, 2)
            }
        
        return outliers
    
    def validate_file(self, filepath: Path) -> Dict:
        """
        Validate a single zip file.
        
        Args:
            filepath: Path to zip file
            
        Returns:
            Dict with complete validation results
        """
        logger.info(f"Validating {filepath.name}...")
        
        try:
            # Read CSV from zip
            with zipfile.ZipFile(filepath, 'r') as z:
                csv_name = [f for f in z.namelist() if f.endswith('.csv')][0]
                with z.open(csv_name) as f:
                    # Read sample for faster validation (first 100k rows)
                    df = pd.read_csv(f, nrows=100000)
            
            # Run validations
            schema_result = self.validate_schema(df, filepath.name)
            missing_values = self.check_missing_values(df)
            outliers = self.detect_outliers(df, schema_result["schema_type"])
            
            result = {
                "filename": filepath.name,
                "file_size_mb": round(filepath.stat().st_size / (1024**2), 2),
                "rows_sampled": len(df),
                "schema": schema_result,
                "missing_values": missing_values,
                "outliers": outliers,
                "validation_timestamp": datetime.now().isoformat()
            }
            
            logger.success(f"Validated {filepath.name}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to validate {filepath.name}: {e}")
            return {
                "filename": filepath.name,
                "error": str(e),
                "validation_timestamp": datetime.now().isoformat()
            }
    
    def validate_all(self) -> List[Dict]:
        """
        Validate all zip files in data directory.
        
        Returns:
            List of validation result dicts
        """
        zip_files = sorted(self.data_dir.glob("*.zip"))
        
        if not zip_files:
            logger.warning(f"No zip files found in {self.data_dir}")
            return []
        
        logger.info(f"Found {len(zip_files)} files to validate")
        
        results = []
        for filepath in zip_files:
            result = self.validate_file(filepath)
            results.append(result)
        
        return results
    
    def generate_report(self, results: List[Dict], output_file: Optional[Path] = None) -> str:
        """
        Generate human-readable validation report.
        
        Args:
            results: List of validation results
            output_file: Optional path to save report
            
        Returns:
            Report as string
        """
        if output_file is None:
            output_file = self.validation_dir / "validation_report.txt"
        
        lines = []
        lines.append("=" * 80)
        lines.append("CITI BIKE DATA VALIDATION REPORT")
        lines.append("=" * 80)
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Files validated: {len(results)}")
        lines.append("")
        
        for result in results:
            if "error" in result:
                lines.append(f"\n[ERROR] {result['filename']}")
                lines.append(f"  Error: {result['error']}")
                continue
            
            lines.append(f"\n{'='*80}")
            lines.append(f"FILE: {result['filename']}")
            lines.append(f"{'='*80}")
            lines.append(f"Size: {result['file_size_mb']} MB")
            lines.append(f"Rows sampled: {result['rows_sampled']:,}")
            
            # Schema info
            schema = result['schema']
            lines.append(f"\nSchema: {schema['schema_type'].upper()}")
            lines.append(f"Valid: {'✓' if schema['valid'] else '✗'}")
            if schema['missing_columns']:
                lines.append(f"Missing columns: {', '.join(schema['missing_columns'])}")
            if schema['extra_columns']:
                lines.append(f"Extra columns: {', '.join(schema['extra_columns'])}")
            
            # Missing values
            lines.append(f"\nMissing Values:")
            missing = result['missing_values']
            significant_missing = {k: v for k, v in missing.items() if v > 0}
            if significant_missing:
                for col, pct in sorted(significant_missing.items(), key=lambda x: x[1], reverse=True):
                    lines.append(f"  {col}: {pct}%")
            else:
                lines.append("  No missing values")
            
            # Outliers
            lines.append(f"\nOutliers:")
            outliers = result['outliers']
            for outlier_type, stats in outliers.items():
                lines.append(f"  {outlier_type.replace('_', ' ').title()}: {stats['count']:,} ({stats['percent']}%)")
        
        # Summary
        lines.append(f"\n{'='*80}")
        lines.append("SUMMARY")
        lines.append(f"{'='*80}")
        
        valid_files = sum(1 for r in results if r.get('schema', {}).get('valid', False))
        error_files = sum(1 for r in results if 'error' in r)
        
        lines.append(f"Valid schema: {valid_files}/{len(results)}")
        lines.append(f"Errors: {error_files}")
        
        report = "\n".join(lines)
        
        # Save to file
        output_file.write_text(report)
        logger.info(f"Validation report saved to {output_file}")
        
        return report


def main():
    """Run validation on all downloaded files."""
    validator = BikeDataValidator()
    
    logger.info("Starting data validation...")
    results = validator.validate_all()
    
    if not results:
        logger.error("No files to validate")
        return
    
    # Generate and display report
    report = validator.generate_report(results)
    print("\n" + report)
    
    logger.success("Validation complete!")


if __name__ == "__main__":
    main()
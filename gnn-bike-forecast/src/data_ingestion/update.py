"""
Incremental Update Manager for NYC Citi Bike Trip Data

Checks for new files on S3 and downloads only new data,
avoiding re-downloading existing files.
"""

from pathlib import Path
from typing import List, Dict, Set
from datetime import datetime

from loguru import logger

from src.data_ingestion.scraper import S3BikeScraper
from src.data_ingestion.download import BikeDataDownloader
from src.data_ingestion.validate import BikeDataValidator


class BikeDataUpdater:
    """Manages incremental updates of Citi Bike data."""
    
    def __init__(self, data_dir: Path = Path("data/raw")):
        """
        Initialize updater.
        
        Args:
            data_dir: Directory containing downloaded files
        """
        self.data_dir = Path(data_dir)
        self.scraper = S3BikeScraper(cache_dir=self.data_dir)
        self.downloader = BikeDataDownloader(data_dir=self.data_dir)
        self.validator = BikeDataValidator(data_dir=self.data_dir)
        logger.info(f"Initialized BikeDataUpdater with data_dir: {self.data_dir}")
    
    def get_local_files(self) -> Set[str]:
        """
        Get set of already downloaded filenames.
        
        Returns:
            Set of filenames (e.g., {"202301-citibike-tripdata.csv.zip", ...})
        """
        local_files = {f.name for f in self.data_dir.glob("*.zip")}
        logger.info(f"Found {len(local_files)} files locally")
        return local_files
    
    def get_remote_files(
        self,
        start_date: str = None,
        end_date: str = None,
        use_cache: bool = False
    ) -> List[Dict[str, str]]:
        """
        Get list of available files from S3.
        
        Args:
            start_date: Optional start date in YYYYMM format
            end_date: Optional end date in YYYYMM format
            use_cache: Whether to use cached S3 listing (default False for updates)
            
        Returns:
            List of file metadata dicts
        """
        remote_files = self.scraper.get_available_files(
            start_date=start_date,
            end_date=end_date,
            use_cache=use_cache
        )
        logger.info(f"Found {len(remote_files)} files on S3")
        return remote_files
    
    def identify_new_files(
        self,
        remote_files: List[Dict[str, str]],
        local_files: Set[str]
    ) -> List[Dict[str, str]]:
        """
        Identify files that exist on S3 but not locally.
        
        Args:
            remote_files: List of file metadata from S3
            local_files: Set of local filenames
            
        Returns:
            List of new file metadata to download
        """
        new_files = [
            f for f in remote_files
            if f["filename"] not in local_files
        ]
        
        logger.info(f"Identified {len(new_files)} new files to download")
        return new_files
    
    def update(
        self,
        start_date: str = None,
        end_date: str = None,
        validate: bool = True,
        max_retries: int = 3
    ) -> Dict[str, int]:
        """
        Check for and download new files from S3.
        
        Args:
            start_date: Optional start date filter (YYYYMM)
            end_date: Optional end date filter (YYYYMM)
            validate: Whether to validate new downloads
            max_retries: Maximum download retry attempts
            
        Returns:
            Dict with update statistics
        """
        logger.info("=" * 60)
        logger.info("Starting incremental update...")
        logger.info("=" * 60)
        
        # Get local and remote files
        local_files = self.get_local_files()
        remote_files = self.get_remote_files(
            start_date=start_date,
            end_date=end_date,
            use_cache=False  # Always fetch fresh listing for updates
        )
        
        # Identify new files
        new_files = self.identify_new_files(remote_files, local_files)
        
        if not new_files:
            logger.info("No new files to download. Data is up to date!")
            return {
                "new_files": 0,
                "downloaded": 0,
                "failed": 0,
                "validated": 0
            }
        
        # Display new files
        logger.info("\nNew files available:")
        for f in new_files:
            size_mb = f["size"] / (1024 ** 2)
            logger.info(f"  - {f['filename']} ({size_mb:.1f} MB)")
        
        # Download new files
        logger.info(f"\nDownloading {len(new_files)} new files...")
        download_results = self.downloader.download_batch(
            files=new_files,
            force=False,
            max_retries=max_retries
        )
        
        # Validate new files
        validated_count = 0
        if validate and download_results["success"] > 0:
            logger.info("\nValidating newly downloaded files...")
            
            # Get list of newly downloaded files
            newly_downloaded = [
                self.data_dir / f["filename"]
                for f in new_files
                if (self.data_dir / f["filename"]).exists()
            ]
            
            # Validate each new file
            validation_results = []
            for filepath in newly_downloaded:
                result = self.validator.validate_file(filepath)
                validation_results.append(result)
                if not result.get("error"):
                    validated_count += 1
            
            # Generate validation report for new files only
            if validation_results:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                report_file = self.validator.validation_dir / f"update_validation_{timestamp}.txt"
                self.validator.generate_report(validation_results, output_file=report_file)
        
        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("Update Summary:")
        logger.info(f"  New files available: {len(new_files)}")
        logger.info(f"  Successfully downloaded: {download_results['success']}")
        logger.info(f"  Failed downloads: {download_results['failed']}")
        if validate:
            logger.info(f"  Validated: {validated_count}")
        logger.info("=" * 60)
        
        return {
            "new_files": len(new_files),
            "downloaded": download_results["success"],
            "failed": download_results["failed"],
            "validated": validated_count
        }
    
    def check_for_updates(
        self,
        start_date: str = None,
        end_date: str = None
    ) -> List[str]:
        """
        Check for new files without downloading.
        
        Args:
            start_date: Optional start date filter (YYYYMM)
            end_date: Optional end date filter (YYYYMM)
            
        Returns:
            List of new filenames available
        """
        local_files = self.get_local_files()
        remote_files = self.get_remote_files(
            start_date=start_date,
            end_date=end_date,
            use_cache=False
        )
        
        new_files = self.identify_new_files(remote_files, local_files)
        
        if new_files:
            logger.info(f"\n{len(new_files)} new files available:")
            for f in new_files:
                logger.info(f"  - {f['filename']}")
        else:
            logger.info("No new files available")
        
        return [f["filename"] for f in new_files]


def main():
    """Check for and download new Citi Bike data."""
    updater = BikeDataUpdater()
    
    # Option 1: Just check for updates (no download)
    logger.info("Checking for new files...")
    new_files = updater.check_for_updates()
    
    if new_files:
        logger.info(f"\nFound {len(new_files)} new files")
        
        # Prompt user (in production, this could be automated)
        response = input("\nDownload new files? (y/n): ").lower().strip()
        
        if response == 'y':
            # Option 2: Download and validate new files
            results = updater.update(validate=True)
            
            if results["failed"] == 0:
                logger.success("Update completed successfully!")
            else:
                logger.warning(f"Update completed with {results['failed']} failures")
        else:
            logger.info("Update cancelled")
    else:
        logger.success("Data is up to date!")


if __name__ == "__main__":
    main()
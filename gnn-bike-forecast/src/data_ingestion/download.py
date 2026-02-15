"""
Download Manager for NYC Citi Bike Trip Data

Downloads monthly trip data files from S3 with progress tracking, error handling, and metadata logging.
"""

import json
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import time

import requests
from tqdm import tqdm
from loguru import logger

from src.data_ingestion.scraper import S3BikeScraper


class BikeDataDownloader:
    """Download and manage Citi Bike trip data files."""
    
    def __init__(self, data_dir: Path = Path("data/raw")):
        """
        Initialize downloader.
        
        Args:
            data_dir: Directory to save downloaded files
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.data_dir / "download_metadata.json"
        self.scraper = S3BikeScraper(cache_dir=self.data_dir)
        logger.info(f"Initialized BikeDataDownloader with data_dir: {self.data_dir}")
    
    def load_metadata(self) -> Dict:
        """
        Load download metadata from JSON file.
        
        Returns:
            Dictionary with download history
        """
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {"downloads": []}
    
    def save_metadata(self, metadata: Dict):
        """
        Save download metadata to JSON file.
        
        Args:
            metadata: Dictionary to save
        """
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.debug(f"Saved metadata to {self.metadata_file}")
    
    def is_downloaded(self, filename: str) -> bool:
        """
        Check if a file has already been downloaded.
        
        Args:
            filename: Name of the file to check
            
        Returns:
            True if file exists in data_dir
        """
        filepath = self.data_dir / filename
        return filepath.exists()
    
    def download_file(
        self, 
        url: str, 
        filename: str, 
        file_size: int,
        force: bool = False
    ) -> bool:
        """
        Download a single file from S3 with progress bar.
        
        Args:
            url: Download URL
            filename: Name to save file as
            file_size: Expected file size in bytes
            force: If True, redownload even if file exists
            
        Returns:
            True if download successful, False otherwise
        """
        filepath = self.data_dir / filename
        
        # Skip if already downloaded
        if not force and self.is_downloaded(filename):
            logger.info(f"Skipping {filename} (already exists)")
            return True
        
        try:
            logger.info(f"Downloading {filename} ({file_size / (1024**2):.1f} MB)...")
            
            # Stream download with progress bar
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()
            
            # Progress bar
            with tqdm(
                total=file_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
                desc=filename
            ) as pbar:
                with open(filepath, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            
            # Verify file size
            actual_size = filepath.stat().st_size
            if actual_size != file_size:
                logger.warning(
                    f"Size mismatch for {filename}: "
                    f"expected {file_size}, got {actual_size}"
                )
            
            logger.success(f"Downloaded {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to download {filename}: {e}")
            # Clean up partial download
            if filepath.exists():
                filepath.unlink()
            return False
    
    def download_batch(
        self,
        files: List[Dict[str, str]],
        force: bool = False,
        max_retries: int = 3
    ) -> Dict[str, int]:
        """
        Download multiple files with retry logic.
        
        Args:
            files: List of file metadata dicts from scraper
            force: If True, redownload existing files
            max_retries: Maximum retry attempts per file
            
        Returns:
            Dict with counts: {"success": X, "failed": Y, "skipped": Z}
        """
        results = {"success": 0, "failed": 0, "skipped": 0}
        metadata = self.load_metadata()
        
        logger.info(f"Starting batch download of {len(files)} files")
        
        for file_info in files:
            filename = file_info["filename"]
            url = file_info["url"]
            size = file_info["size"]
            
            # Skip if already downloaded
            if not force and self.is_downloaded(filename):
                results["skipped"] += 1
                continue
            
            # Retry logic
            success = False
            for attempt in range(1, max_retries + 1):
                if attempt > 1:
                    logger.info(f"Retry {attempt}/{max_retries} for {filename}")
                    time.sleep(2 ** attempt)  # Exponential backoff
                
                success = self.download_file(url, filename, size, force=force)
                if success:
                    break
            
            # Update results and metadata
            if success:
                results["success"] += 1
                metadata["downloads"].append({
                    "filename": filename,
                    "url": url,
                    "size": size,
                    "download_timestamp": datetime.now().isoformat(),
                    "last_modified": file_info["last_modified"]
                })
            else:
                results["failed"] += 1
        
        # Save metadata
        self.save_metadata(metadata)
        
        # Summary
        logger.info("=" * 60)
        logger.info("Download Summary:")
        logger.info(f"  Success: {results['success']}")
        logger.info(f"  Failed:  {results['failed']}")
        logger.info(f"  Skipped: {results['skipped']}")
        logger.info("=" * 60)
        
        return results
    
    def download_date_range(
        self,
        start_date: str,
        end_date: str,
        force: bool = False
    ) -> Dict[str, int]:
        """
        Download all files within a date range.
        
        Args:
            start_date: Start date in YYYYMM format
            end_date: End date in YYYYMM format
            force: If True, redownload existing files
            
        Returns:
            Download results dict
        """
        logger.info(f"Fetching files for date range: {start_date} to {end_date}")
        files = self.scraper.get_available_files(
            start_date=start_date,
            end_date=end_date,
            use_cache=True
        )
        
        if not files:
            logger.warning("No files found for the specified date range")
            return {"success": 0, "failed": 0, "skipped": 0}
        
        logger.info(f"Found {len(files)} files to download")
        return self.download_batch(files, force=force)
    
    def get_downloaded_files(self) -> List[str]:
        """
        Get list of already downloaded files.
        
        Returns:
            List of filenames
        """
        return [f.name for f in self.data_dir.glob("*.zip")]


def main():
    """Download 6 months of data (Jan-Jun 2024)."""
    downloader = BikeDataDownloader()
    
    logger.info("Starting download of Jan-Jun 2024 data...")
    results = downloader.download_date_range(
        start_date="202401",
        end_date="202406",
        force=False  # Skip already downloaded files
    )
    
    if results["failed"] > 0:
        logger.error("Some downloads failed. Check logs above.")
    else:
        logger.success("All downloads completed successfully!")
    
    # List downloaded files
    downloaded = downloader.get_downloaded_files()
    logger.info(f"\nDownloaded files in {downloader.data_dir}:")
    for f in sorted(downloaded):
        logger.info(f"  - {f}")


if __name__ == "__main__":
    main()
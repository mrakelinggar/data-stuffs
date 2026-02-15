import re 
from typing import List, Dict, Optional 
from datetime import datetime 
from pathlib import Path

import requests 
from bs4 import BeautifulSoup 
from loguru import logger

class S3BikeScraper:
    """Scraper for Citi Bike Public S3 bucket"""
    
    S3_BUCKET_URL = "https://s3.amazonaws.com/tripdata"
    NYC_PATTERN = r"^\d{6}-citibike-tripdata\.zip$"  # YYYYMM-citibike-tripdata.csv.zip
    JC_PATTERN = r"^JC-"  # Jersey City prefix to exclude
    
    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or Path("data/raw")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initialized S3BikeScrapper with cache directory: {self.cache_dir}")

    def fetch_bucket_listing(self, use_cache: bool = True) -> str:
        """
        Fetch S3 bucket XML listing.
        
        Args:
            use_cache: If True, use cached listing if available (valid for 1 hour)
            
        Returns:
            XML content as string
        """
        cache_file = self.cache_dir / "s3_listing_cache.xml"
        
        # Check cache
        if use_cache and cache_file.exists():
            age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
            if age.total_seconds() < 3600:  # Cache valid for 1 hour
                logger.info(f"Using cached listing (age: {age.total_seconds():.0f}s)")
                return cache_file.read_text()
        
        # Fetch from S3
        logger.info(f"Fetching bucket listing from {self.S3_BUCKET_URL}")
        response = requests.get(self.S3_BUCKET_URL, timeout=30)
        response.raise_for_status()
        
        # Cache the response
        cache_file.write_text(response.text)
        logger.info(f"Cached listing to {cache_file}")
        
        return response.text
    
    def parse_listing(self, xml_content: str) -> List[Dict[str, str]]:
        """
        Parse S3 XML listing to extract file metadata.
        
        Args:
            xml_content: Raw XML string from S3
            
        Returns:
            List of dicts with keys: filename, url, size, last_modified
        """
        soup = BeautifulSoup(xml_content, "lxml-xml")
        contents = soup.find_all("Contents")
        
        files = []
        for content in contents:
            key = content.find("Key")
            size = content.find("Size")
            last_modified = content.find("LastModified")
            
            if key and size and last_modified:
                filename = key.text
                files.append({
                    "filename": filename,
                    "url": f"{self.S3_BUCKET_URL}/{filename}",
                    "size": int(size.text),
                    "last_modified": last_modified.text
                })
        
        logger.info(f"Parsed {len(files)} files from listing")
        return files
    
    def filter_nyc_files(self, files: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Filter for NYC Citi Bike monthly data files.
        
        Includes: YYYYMM-citibike-tripdata.csv.zip
        Excludes: Jersey City files (JC- prefix)
        
        Args:
            files: List of file metadata dicts
            
        Returns:
            Filtered list containing only NYC monthly trip data
        """
        nyc_pattern = re.compile(self.NYC_PATTERN)
        jc_pattern = re.compile(self.JC_PATTERN)
        
        filtered = [
            f for f in files
            if nyc_pattern.match(f["filename"]) and not jc_pattern.match(f["filename"])
        ]
        
        logger.info(f"Filtered to {len(filtered)} NYC files (from {len(files)} total)")
        return sorted(filtered, key=lambda x: x["filename"])
    
    def get_available_files(
            self, 
            start_date: Optional[str] = None,
            end_date: Optional[str] = None,
            use_cache: bool = True,
    ) -> List[Dict[str, str]]:
        """
        Get list of available NYC Citi Bike trip data files within date range.
        
        Args:
            start_date: Optional start date (YYYY-MM) to filter files
            end_date: Optional end date (YYYY-MM) to filter files
            use_cache: If True, use cached listing if available
            
        Returns:
            List of dicts with file metadata for available NYC trip data files
        """
        xml_content = self.fetch_bucket_listing(use_cache=use_cache)
        all_files = self.parse_listing(xml_content)
        nyc_files = self.filter_nyc_files(all_files)

        if len(nyc_files) == 0:
            logger.error("No NYC files found for the specified criteria.")
        
        # Date filtering
        if start_date or end_date:
            filtered = []
            for f in nyc_files:
                month = f["filename"][:6]  # Extract YYYYMM
                
                if start_date and month < start_date:
                    continue
                if end_date and month > end_date:
                    continue
                    
                filtered.append(f)
            
            logger.info(
                f"Date filtered: {len(filtered)} files "
                f"(range: {start_date or 'start'} to {end_date or 'end'})"
            )
            return filtered
        


        return nyc_files

    def get_file_info(self, filename: str) -> Optional[Dict[str, str]]:
        """
        Get metadata for a specific file.
        
        Args:
            filename: Name of the file (e.g., "202301-citibike-tripdata.csv.zip")
            
        Returns:
            File metadata dict or None if not found
        """
        files = self.get_available_files()
        for f in files:
            if f["filename"] == filename:
                return f
        return None


def main():
    """Demo usage of S3BikeScraper."""
    scraper = S3BikeScraper()
    
    # Get all available files
    logger.info("=" * 60)
    logger.info("Fetching all available NYC Citi Bike files...")
    all_files = scraper.get_available_files()
    
    logger.info(f"\nFound {len(all_files)} files")
    logger.info(f"Date range: {all_files[0]['filename'][:6]} to {all_files[-1]['filename'][:6]}")
    
    # Get files for specific date range (Jan-Jun 2023)
    logger.info("\n" + "=" * 60)
    logger.info("Fetching files for Jan-Jun 2024...")
    target_files = scraper.get_available_files(start_date="202401", end_date="202406")
    
    logger.info(f"\nFound {len(target_files)} files for the period:")
    for f in target_files:
        size_mb = f["size"] / (1024 * 1024)
        logger.info(f"  - {f['filename']} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()

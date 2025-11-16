"""Data source MCP server implementation."""

import os
import logging
import pandas as pd
from typing import List, Dict, Any, Optional

from ..config import get_settings
from .data_loader import load_raw_transactions_local

# Try to import kagglehub
KAGGLEHUB_LOADED = False
try:
    import kagglehub
    KAGGLEHUB_LOADED = True
except ImportError:
    pass


class DataSourceMCP:
    """Data source MCP server for data ingestion."""
    
    def __init__(self, dataset_id: str = None, 
                 csv_file_path: Optional[str] = None, 
                 max_rows: Optional[int] = None):
        settings = get_settings()
        self.dataset_id = dataset_id or settings.DEFAULT_DATASET_ID
        self.csv_file_path = csv_file_path
        self.max_rows = max_rows
        self.logger = logging.getLogger("DataSourceServer")
        
    def download_and_load_data(self) -> List[Dict[str, Any]]:
        """Downloads the Kaggle dataset via kagglehub and loads data.
        If csv_file_path is provided, loads from local file instead.
        If max_rows is provided, limits the number of rows processed."""
        
        # If local CSV file is provided, use it instead of Kaggle
        if self.csv_file_path:
            return load_raw_transactions_local(self.csv_file_path, self.max_rows)
        
        settings = get_settings()
        FILE_NAME = settings.DEFAULT_CSV_FILENAME
        
        if not KAGGLEHUB_LOADED:
            self.logger.warning("Kagglehub not available. Falling back to local file load.")
            return [{"Error": "Kagglehub not loaded and no local file provided for simulation."}]

        try:
            self.logger.info(f"Downloading/fetching dataset {self.dataset_id} using kagglehub...")
            
            # Use kagglehub to download the latest version and get the local path
            path_to_files = kagglehub.dataset_download(self.dataset_id)
            self.logger.info(f"Dataset files are located at: {path_to_files}")
            
            file_path = os.path.join(path_to_files, FILE_NAME)
            
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Expected file {FILE_NAME} not found in {path_to_files}")
            
            # Check file size first to determine if we should use chunking
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            self.logger.info(f"CSV file size: {file_size_mb:.2f} MB")
            
            # For large files, use chunked reading to avoid memory issues
            if self.max_rows is not None:
                # If max_rows is specified, read only that many rows
                df = pd.read_csv(file_path, nrows=self.max_rows)
                self.logger.info(f"Limiting to first {self.max_rows} rows as specified.")
            else:
                # For very large files (>100MB), warn and suggest using max_rows
                if file_size_mb > 100:
                    self.logger.warning(f"Large file detected ({file_size_mb:.2f} MB). This may cause memory issues.")
                    self.logger.warning("Consider using --max-rows to limit the number of rows processed.")
                    self.logger.warning("Attempting to load full file...")
                
                # Try to read the file in chunks if it's very large
                try:
                    # First, get the total row count efficiently
                    chunk_iter = pd.read_csv(file_path, chunksize=10000)
                    total_rows = 0
                    for chunk in chunk_iter:
                        total_rows += len(chunk)
                    
                    self.logger.info(f"Total rows in CSV: {total_rows:,}")
                    
                    # If file is very large, read in chunks and combine
                    if total_rows > 1000000:  # More than 1 million rows
                        self.logger.info("Large dataset detected. Reading in chunks...")
                        chunks = []
                        chunk_iter = pd.read_csv(file_path, chunksize=50000)
                        for i, chunk in enumerate(chunk_iter):
                            chunks.append(chunk.fillna(0))
                            if (i + 1) % 10 == 0:
                                self.logger.info(f"Processed {(i + 1) * 50000:,} rows...")
                        df = pd.concat(chunks, ignore_index=True)
                        self.logger.info(f"Successfully loaded all {len(df):,} rows in chunks.")
                    else:
                        # For smaller files, read normally
                        df = pd.read_csv(file_path)
                        self.logger.info(f"Processing all {len(df):,} rows from the CSV.")
                except MemoryError:
                    error_msg = (
                        f"Memory error loading file. The dataset has {total_rows:,} rows. "
                        f"Please use --max-rows to limit the number of rows (e.g., --max-rows 100000)."
                    )
                    raise MemoryError(error_msg)
            
            # Fill NaN values and convert to dict
            processed_df = df.fillna(0)
            data_list = processed_df.to_dict('records')
            
            self.logger.info(f"Successfully loaded {len(data_list):,} records from {file_path}.")
            return data_list
            
        except MemoryError as e:
            self.logger.error(f"Memory error: {e}")
            return [{"Error": f"Memory error loading dataset. The file is too large to load entirely into memory. Please use --max-rows to limit processing (e.g., --max-rows 100000 or --max-rows 500000)."}]
        except Exception as e:
            self.logger.error(f"Error fetching data using kagglehub: {e}")
            return [{"Error": f"Kagglehub error: {e}"}]




"""Data loading utilities."""

import os
import logging
import pandas as pd
from typing import List, Dict, Any, Optional


def load_raw_transactions_local(file_path: str, max_rows: Optional[int] = None) -> List[Dict[str, Any]]:
    """Loads transaction data from a local CSV file."""
    logger = logging.getLogger("DataLoader")
    
    try:
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            abs_path = os.path.abspath(file_path)
            error_msg = f"File not found: {file_path}"
            if not os.path.isabs(file_path):
                error_msg += f"\nTried absolute path: {abs_path}"
            error_msg += "\nPlease provide the full path to your CSV file."
            return [{"Error": error_msg}]
        
        logger.info(f"Loading data from local CSV file: {file_path}")
        
        # Check file size first to determine if we should use chunking
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        logger.info(f"CSV file size: {file_size_mb:.2f} MB")
        
        # For large files, use chunked reading to avoid memory issues
        if max_rows is not None:
            # If max_rows is specified, read only that many rows
            df = pd.read_csv(file_path, nrows=max_rows)
            logger.info(f"Limiting to first {max_rows} rows as specified.")
        else:
            # For very large files (>100MB), warn and suggest using max_rows
            if file_size_mb > 100:
                logger.warning(f"Large file detected ({file_size_mb:.2f} MB). This may cause memory issues.")
                logger.warning("Consider using --max-rows to limit the number of rows processed.")
                logger.warning("Attempting to load full file...")
            
            # Try to read the file in chunks if it's very large
            try:
                # First, get the total row count efficiently
                chunk_iter = pd.read_csv(file_path, chunksize=10000)
                total_rows = 0
                for chunk in chunk_iter:
                    total_rows += len(chunk)
                
                logger.info(f"Total rows in CSV: {total_rows:,}")
                
                # If file is very large, read in chunks and combine
                if total_rows > 1000000:  # More than 1 million rows
                    logger.info("Large dataset detected. Reading in chunks...")
                    chunks = []
                    chunk_iter = pd.read_csv(file_path, chunksize=50000)
                    for i, chunk in enumerate(chunk_iter):
                        chunks.append(chunk.fillna(0))
                        if (i + 1) % 10 == 0:
                            logger.info(f"Processed {(i + 1) * 50000:,} rows...")
                    df = pd.concat(chunks, ignore_index=True)
                    logger.info(f"Successfully loaded all {len(df):,} rows in chunks.")
                else:
                    # For smaller files, read normally
                    df = pd.read_csv(file_path)
                    logger.info(f"Processing all {len(df):,} rows from the CSV.")
            except MemoryError:
                error_msg = (
                    f"Memory error loading file. The dataset has {total_rows:,} rows. "
                    f"Please use --max-rows to limit the number of rows (e.g., --max-rows 100000)."
                )
                raise MemoryError(error_msg)
        
        # Fill NaN values and convert to dict
        processed_df = df.fillna(0)
        data_list = processed_df.to_dict('records')
        
        logger.info(f"Successfully loaded {len(data_list):,} records from {file_path}.")
        return data_list
        
    except MemoryError as e:
        logger.error(f"Memory error: {e}")
        return [{"Error": f"Memory error loading file. The file is too large to load entirely into memory. Please use --max-rows to limit processing (e.g., --max-rows 100000 or --max-rows 500000)."}]
    except Exception as e:
        logger.error(f"Error loading local CSV file: {e}")
        return [{"Error": f"Error loading CSV: {e}"}]




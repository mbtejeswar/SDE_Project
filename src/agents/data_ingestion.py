"""Data Ingestion Agent."""

import logging
from typing import List, Dict, Any, Optional

from ..data import DataSourceMCP
from ..mcp.clients import ExternalDataSourceMCPClient
from ..config import get_settings


class DataIngestionAgent:
    """Agent responsible for ingesting raw banking data."""
    
    def __init__(self):
        self.logger = logging.getLogger("DataIngestionAgent")
        self.settings = get_settings()
        self.external_client = None
        
        if self.settings.use_external_data_source:
            try:
                self.external_client = ExternalDataSourceMCPClient(
                    self.settings.EXTERNAL_DATA_SOURCE_MCP_URL
                )
            except Exception as e:
                self.logger.warning(f"Failed to create external Data Source client: {e}")
    
    def fetch_raw_data(
        self, 
        csv_file_path: Optional[str] = None,
        max_rows: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Fetch raw banking data.
        
        Args:
            csv_file_path: Optional path to local CSV file
            max_rows: Optional limit on number of rows to process
            
        Returns:
            List of raw transaction records
        """
        if self.settings.use_external_data_source and self.external_client:
            self.logger.info("Loading data using external Data Source MCP server...")
            return self.external_client.fetch_raw_banking_data(
                csv_file_path=csv_file_path,
                max_rows=max_rows
            )
        else:
            self.logger.info("Loading data using local DataSourceMCP...")
            ds_server = DataSourceMCP(csv_file_path=csv_file_path, max_rows=max_rows)
            return ds_server.download_and_load_data()




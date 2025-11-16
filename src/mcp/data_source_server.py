"""Data Source MCP server implementation."""

import sys
import logging
from typing import List, Dict, Any, Optional

from ..config import get_settings
from ..data import DataSourceMCP

# Try to import MCP server
MCP_SERVER_AVAILABLE = False
USE_FASTMCP = False
USE_STANDARD_MCP = False

try:
    from fastmcp import FastMCP
    MCP_SERVER_AVAILABLE = True
    USE_FASTMCP = True
except ImportError:
    try:
        from mcp.server import Server
        from mcp.server.stdio import stdio_server
        from mcp.types import Tool, TextContent
        import asyncio
        MCP_SERVER_AVAILABLE = True
        USE_STANDARD_MCP = True
        USE_FASTMCP = False
    except ImportError:
        # Fallback placeholder
        class FastMCP:
            def __init__(self, name):
                self.name = name
                self.tools = []
            def tool(self):
                return lambda f: f
            def run(self, transport='stdio'):
                print(f"--- {self.name} MCP Server Running via {transport} ---")


def create_data_source_mcp_server():
    """Create and configure the Data Source MCP server."""
    settings = get_settings()
    
    if settings.use_external_data_source:
        return None
    
    if not MCP_SERVER_AVAILABLE:
        return None
    
    if USE_FASTMCP:
        data_source_mcp = FastMCP("DataSource")
        
        @data_source_mcp.tool()
        def fetch_raw_banking_data(
            source_identifier: str = None,
            csv_file_path: Optional[str] = None,
            max_rows: Optional[int] = None
        ) -> List[Dict[str, Any]]:
            """Tool exposed to the Data Ingestion Agent to fetch raw transaction data.
            
            Args:
                source_identifier: Kaggle dataset ID (if using Kaggle)
                csv_file_path: Path to local CSV file (if provided, overrides Kaggle)
                max_rows: Maximum number of rows to process (None = all rows)
            """
            dataset_id = source_identifier or settings.DEFAULT_DATASET_ID
            ds_server = DataSourceMCP(
                dataset_id=dataset_id, 
                csv_file_path=csv_file_path,
                max_rows=max_rows
            )
            return ds_server.download_and_load_data()
        
        return data_source_mcp
    
    return None


def run_data_source_mcp_server():
    """Run the Data Source MCP server as a standalone service."""
    data_source_mcp = create_data_source_mcp_server()
    
    if data_source_mcp:
        print("Starting Data Source MCP Server...", file=sys.stderr)
        data_source_mcp.run(transport='stdio')
    else:
        print("ERROR: MCP not available. Install with: pip install fastmcp", file=sys.stderr)
        sys.exit(1)




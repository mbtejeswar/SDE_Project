"""External MCP client implementations."""

import logging
from typing import Dict, Any, Optional

# Try to import requests
REQUESTS_AVAILABLE = False
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    pass

# Try to import MCP client
MCP_CLIENT_AVAILABLE = False
try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    import asyncio
    MCP_CLIENT_AVAILABLE = True
except ImportError:
    try:
        from mcp.client import ClientSession
        from mcp.client.stdio import stdio_client
        import asyncio
        MCP_CLIENT_AVAILABLE = True
    except ImportError:
        pass


class ExternalDataSourceMCPClient:
    """Client to connect to external Data Source MCP server."""
    
    def __init__(self, server_url: str):
        self.server_url = server_url
        self.logger = logging.getLogger("ExternalDataSourceClient")
        self.session = None
        
    async def connect(self):
        """Connect to external MCP server."""
        if not MCP_CLIENT_AVAILABLE:
            raise RuntimeError("MCP client not available. Install with: pip install mcp")
        
        self.logger.info(f"Connecting to external Data Source MCP server at {self.server_url}")
        
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Call a tool on the external MCP server."""
        if not self.session:
            await self.connect()
        
        raise NotImplementedError("External MCP client implementation depends on server transport type")
    
    def fetch_raw_banking_data(self, source_identifier: str = None, 
                               csv_file_path: Optional[str] = None,
                               max_rows: Optional[int] = None) -> list:
        """Fetch raw banking data from external MCP server."""
        if not REQUESTS_AVAILABLE:
            self.logger.error("requests library not available. Install with: pip install requests")
            return [{"Error": "External MCP client requires 'requests' library"}]
        
        try:
            response = requests.post(
                f"{self.server_url}/tools/fetch_raw_banking_data",
                json={
                    "source_identifier": source_identifier,
                    "csv_file_path": csv_file_path,
                    "max_rows": max_rows
                },
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error calling external MCP server: {e}")
            return [{"Error": f"External MCP server error: {str(e)}"}]
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")
            return [{"Error": f"Unexpected error: {str(e)}"}]


class ExternalFraudInferenceMCPClient:
    """Client to connect to external Fraud Inference MCP server."""
    
    def __init__(self, server_url: str):
        self.server_url = server_url
        self.logger = logging.getLogger("ExternalFraudInferenceClient")
        self.session = None
        
    async def connect(self):
        """Connect to external MCP server."""
        if not MCP_CLIENT_AVAILABLE:
            raise RuntimeError("MCP client not available. Install with: pip install mcp")
        
        self.logger.info(f"Connecting to external Fraud Inference MCP server at {self.server_url}")
        
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Call a tool on the external MCP server."""
        if not self.session:
            await self.connect()
        
        raise NotImplementedError("External MCP client implementation depends on server transport type")
    
    def predict_fraud_score(self, transaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict fraud score using external MCP server."""
        import json
        if not REQUESTS_AVAILABLE:
            self.logger.error("requests library not available. Install with: pip install requests")
            return {"Error": "External MCP client requires 'requests' library"}
        
        try:
            response = requests.post(
                f"{self.server_url}/tools/predict_fraud_score",
                json={"transaction_data": json.dumps(transaction_data)},
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error calling external MCP server: {e}")
            return {"Error": f"External MCP server error: {str(e)}"}
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")
            return {"Error": f"Unexpected error: {str(e)}"}
    
    def explain_fraud_prediction(self, transaction_data: Dict[str, Any]) -> Optional[str]:
        """Explain fraud prediction using external MCP server."""
        import json
        if not REQUESTS_AVAILABLE:
            self.logger.error("requests library not available. Install with: pip install requests")
            return "Error: External MCP client requires 'requests' library"
        
        try:
            response = requests.post(
                f"{self.server_url}/tools/explain_fraud_prediction",
                json={"transaction_data": json.dumps(transaction_data)},
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            return result.get("explanation") or result.get("result")
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error calling external MCP server: {e}")
            return f"Error: External MCP server error: {str(e)}"
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")
            return f"Error: Unexpected error: {str(e)}"




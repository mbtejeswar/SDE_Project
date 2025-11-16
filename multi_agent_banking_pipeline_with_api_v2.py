import pandas as pd
import logging
from typing import List, Dict, Any, Optional
import os
import sys
import zipfile 
import io
import argparse
import secrets
from datetime import datetime

# --- Requests for External MCP Client ---
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    print("WARNING: 'requests' library not installed. External MCP clients will not work.")
    print("Install with: pip install requests")

# --- FastAPI Import for Web API Server ---
FASTAPI_AVAILABLE = False
try:
    from fastapi import FastAPI, HTTPException, Depends, Header, status
    from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    print("WARNING: 'fastapi' and 'uvicorn' not installed. Web API server will not be available.")
    print("Install with: pip install fastapi uvicorn") 

# --- New Import: kagglehub ---
KAGGLEHUB_LOADED = False
try:
    import kagglehub
    KAGGLEHUB_LOADED = True
except ImportError:
    print("WARNING: 'kagglehub' not installed. Data ingestion will use local file simulation.")
    
# --- MCP Framework Import ---
# Try to import MCP client for external server connections
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
        print("WARNING: MCP client not available. External MCP servers will not work.")
        print("Install with: pip install mcp")

# Try to import MCP server for local server creation (fallback)
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
        print("WARNING: Neither 'fastmcp' nor 'mcp' package installed.")
        print("Install with: pip install fastmcp  OR  pip install mcp")
        MCP_SERVER_AVAILABLE = False
        USE_STANDARD_MCP = False
        USE_FASTMCP = False
        # Fallback placeholder
        class FastMCP:
            def __init__(self, name):
                self.name = name
                self.tools = []
            def tool(self):
                return lambda f: f
            def run(self, transport='stdio'):
                print(f"--- {self.name} MCP Server Running via {transport} ---")
else:
    USE_STANDARD_MCP = False
    USE_FASTMCP = True

# --- External MCP Server Configuration ---
# Get external MCP server URLs from environment variables
EXTERNAL_DATA_SOURCE_MCP_URL = os.getenv("EXTERNAL_DATA_SOURCE_MCP_URL", None)
EXTERNAL_FRAUD_INFERENCE_MCP_URL = os.getenv("EXTERNAL_FRAUD_INFERENCE_MCP_URL", None)

# Use external servers if URLs are provided, otherwise use local
USE_EXTERNAL_DATA_SOURCE = EXTERNAL_DATA_SOURCE_MCP_URL is not None
USE_EXTERNAL_FRAUD_INFERENCE = EXTERNAL_FRAUD_INFERENCE_MCP_URL is not None

if USE_EXTERNAL_DATA_SOURCE:
    print(f"INFO: Using external Data Source MCP server: {EXTERNAL_DATA_SOURCE_MCP_URL}")
if USE_EXTERNAL_FRAUD_INFERENCE:
    print(f"INFO: Using external Fraud Inference MCP server: {EXTERNAL_FRAUD_INFERENCE_MCP_URL}")
        
# --- Shared Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
EXPECTED_FEATURES = [
    'Pseudo_Transaction_ID', 'type', 'amount', 'oldbalanceOrg', 
    'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'nameOrig'
]

# =================================================================
# ## 1. DATA SOURCE MCP SERVER (for Data Ingestion Agent)
# =================================================================

# External MCP Client for Data Source
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
        
        # For HTTP/SSE transport, you would use:
        # from mcp.client.sse import sse_client
        # async with sse_client(url=self.server_url) as (read, write):
        #     async with ClientSession(read, write) as session:
        #         self.session = session
        
        # For stdio transport (if external server is accessible via command):
        # This is a placeholder - actual implementation depends on your external server setup
        self.logger.info(f"Connecting to external Data Source MCP server at {self.server_url}")
        
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Call a tool on the external MCP server."""
        if not self.session:
            await self.connect()
        
        # Placeholder - actual implementation depends on MCP client API
        # result = await self.session.call_tool(tool_name, arguments)
        # return result
        raise NotImplementedError("External MCP client implementation depends on server transport type")
    
    def fetch_raw_banking_data(self, source_identifier: str = None, 
                               csv_file_path: Optional[str] = None,
                               max_rows: Optional[int] = None) -> List[Dict[str, Any]]:
        """Fetch raw banking data from external MCP server."""
        if not REQUESTS_AVAILABLE:
            self.logger.error("requests library not available. Install with: pip install requests")
            return [{"Error": "External MCP client requires 'requests' library"}]
        
        try:
            # For HTTP-based external servers, use requests:
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

# Create local FastMCP instance for DataSource (fallback if not using external)
if not USE_EXTERNAL_DATA_SOURCE and MCP_SERVER_AVAILABLE and USE_FASTMCP:
    data_source_mcp = FastMCP("DataSource")
elif not USE_EXTERNAL_DATA_SOURCE:
    data_source_mcp = None
else:
    data_source_mcp = None  # Using external, no local server needed

# Create external client if URL is provided
external_data_source_client = None
if USE_EXTERNAL_DATA_SOURCE:
    try:
        external_data_source_client = ExternalDataSourceMCPClient(EXTERNAL_DATA_SOURCE_MCP_URL)
    except Exception as e:
        print(f"WARNING: Failed to create external Data Source client: {e}")
        USE_EXTERNAL_DATA_SOURCE = False

class DataSourceMCP:
    def __init__(self, dataset_id: str = 'rupakroy/online-payments-fraud-detection-dataset', 
                 csv_file_path: Optional[str] = None, max_rows: Optional[int] = None):
        self.dataset_id = dataset_id
        self.csv_file_path = csv_file_path
        self.max_rows = max_rows
        self.logger = logging.getLogger("DataSourceServer")
        
    def download_and_load_data(self) -> List[Dict[str, Any]]:
        """Downloads the Kaggle dataset via kagglehub and loads data.
        If csv_file_path is provided, loads from local file instead.
        If max_rows is provided, limits the number of rows processed."""
        
        # If local CSV file is provided, use it instead of Kaggle
        if self.csv_file_path:
            return self.load_raw_transactions_local(self.csv_file_path)
        
        # Define the exact file name within the dataset's directory
        FILE_NAME = 'PS_20174392719_1491204439457_log.csv'
        
        if not KAGGLEHUB_LOADED:
            self.logger.warning("Kagglehub not available. Falling back to local file load.")
            # Since we cannot run a local file, we return an error for clarity in a real environment
            # In a local test environment, this would try to load 'PS_20174392719_1491204439457_log.csv'
            return [{"Error": "Kagglehub not loaded and no local file provided for simulation."}]

        try:
            self.logger.info(f"Downloading/fetching dataset {self.dataset_id} using kagglehub...")
            
            # Use kagglehub to download the latest version and get the local path
            # NOTE: This step requires 'kagglehub' installed and credentials configured.
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

    def load_raw_transactions_local(self, file_path: str) -> List[Dict[str, Any]]:
        """Loads transaction data from a local CSV file."""
        try:
            if not os.path.exists(file_path):
                self.logger.error(f"File not found: {file_path}")
                # Check if it's a relative path and suggest absolute path
                abs_path = os.path.abspath(file_path)
                error_msg = f"File not found: {file_path}"
                if not os.path.isabs(file_path):
                    error_msg += f"\nTried absolute path: {abs_path}"
                error_msg += "\nPlease provide the full path to your CSV file."
                return [{"Error": error_msg}]
            
            self.logger.info(f"Loading data from local CSV file: {file_path}")
            
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
            return [{"Error": f"Memory error loading file. The file is too large to load entirely into memory. Please use --max-rows to limit processing (e.g., --max-rows 100000 or --max-rows 500000)."}]
        except Exception as e:
            self.logger.error(f"Error loading local CSV file: {e}")
            return [{"Error": f"Error loading CSV: {e}"}]

# MCP Tool for DataSource
if data_source_mcp:
    @data_source_mcp.tool()
    def fetch_raw_banking_data(source_identifier: str = 'rupakroy/online-payments-fraud-detection-dataset',
                               csv_file_path: Optional[str] = None,
                               max_rows: Optional[int] = None) -> List[Dict[str, Any]]:
        """Tool exposed to the Data Ingestion Agent to fetch raw transaction data.
        
        Args:
            source_identifier: Kaggle dataset ID (if using Kaggle)
            csv_file_path: Path to local CSV file (if provided, overrides Kaggle)
            max_rows: Maximum number of rows to process (None = all rows)
        """
        ds_server = DataSourceMCP(dataset_id=source_identifier, 
                                 csv_file_path=csv_file_path,
                                 max_rows=max_rows)
        return ds_server.download_and_load_data()

# =================================================================
# ## 2. FRAUD INFERENCE MCP SERVER (for Fraud Inference Agent)
# =================================================================

# External MCP Client for Fraud Inference
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

class FraudDetectorModel:
    """Simulated ML Model."""
    def __init__(self):
        # Set the threshold to 0.5000 as requested
        self.FRAUD_THRESHOLD = 0.50 

    def predict_score(self, transaction: Dict[str, Any]) -> float:
        transaction_type = transaction.get('type', 'UNKNOWN')
        amount = transaction.get('amount', 0.0)
        old_orig_bal = transaction.get('oldbalanceOrg', 0.0)
        new_orig_bal = transaction.get('newbalanceOrig', 0.0)
        score = 0.0
        
        # Base score for high-risk types
        if transaction_type in ['TRANSFER', 'CASH_OUT']:
            score += 0.5
            # Bonus 1: Large amount transfer without fully emptying account
            if amount > 100000 and amount <= old_orig_bal:
                score += 0.3
            # Bonus 2: Account drained (characteristic of fraud)
            if old_orig_bal > 0.0 and new_orig_bal == 0.0 and amount >= old_orig_bal:
                score += 0.2
        return min(max(score, 0.05), 1.0) 

    def get_feature_importance(self, transaction: Dict[str, Any], score: float) -> List[Dict[str, Any]]:
        if score < self.FRAUD_THRESHOLD: return []
        impacts = []
        if transaction.get('type') in ['TRANSFER', 'CASH_OUT']:
            impacts.append({"feature": "type", "impact": "High (Common fraud vector)"})
        if transaction.get('amount', 0) > 100000:
            impacts.append({"feature": "amount", "impact": "High (Value is unusually large)"})
        if transaction.get('oldbalanceOrg', 0) > 0.0 and transaction.get('newbalanceOrig', 0) == 0.0:
            impacts.append({"feature": "balance depletion", "impact": "Critical (Originator's account was drained)"})
        return impacts

# Create local FastMCP instance for Fraud Inference (fallback if not using external)
if not USE_EXTERNAL_FRAUD_INFERENCE and MCP_SERVER_AVAILABLE and USE_FASTMCP:
    fraud_inference_mcp = FastMCP("FraudInference")
elif not USE_EXTERNAL_FRAUD_INFERENCE:
    fraud_inference_mcp = None
else:
    fraud_inference_mcp = None  # Using external, no local server needed

# Create external client if URL is provided
external_fraud_inference_client = None
if USE_EXTERNAL_FRAUD_INFERENCE:
    try:
        external_fraud_inference_client = ExternalFraudInferenceMCPClient(EXTERNAL_FRAUD_INFERENCE_MCP_URL)
    except Exception as e:
        print(f"WARNING: Failed to create external Fraud Inference client: {e}")
        USE_EXTERNAL_FRAUD_INFERENCE = False

# Global fraud model instance for MCP tools
fraud_model_instance = FraudDetectorModel()
fraud_logger = logging.getLogger("FraudInferenceServer")

class FraudInferenceMCP:
    def __init__(self):
        self.logger = logging.getLogger("FraudInferenceServer")
        self.fraud_model = FraudDetectorModel()

# MCP Tools for Fraud Inference
if fraud_inference_mcp:
    @fraud_inference_mcp.tool()
    def predict_score(transaction: str) -> float:
        """MCP tool to predict fraud score for a transaction. Accepts JSON string."""
        import json
        try:
            if isinstance(transaction, str):
                transaction_dict = json.loads(transaction)
            else:
                transaction_dict = transaction
            score = fraud_model_instance.predict_score(transaction_dict)
            transaction_id = transaction_dict.get('Pseudo_Transaction_ID', 'N/A')
            fraud_logger.info(f"Predicted score {score:.4f} for transaction {transaction_id}")
            return score
        except (json.JSONDecodeError, TypeError) as e:
            fraud_logger.error(f"Error parsing transaction data: {e}")
            return 0.0
    
    @fraud_inference_mcp.tool()
    def predict_fraud_score(transaction_data: str) -> Dict[str, Any]:
        """MCP tool to predict fraud score and flag status. Accepts JSON string."""
        import json
        try:
            if isinstance(transaction_data, str):
                transaction_dict = json.loads(transaction_data)
            else:
                transaction_dict = transaction_data
            
            transaction_id = transaction_dict.get('Pseudo_Transaction_ID', 'N/A')
            
            if not all(feature in transaction_dict for feature in EXPECTED_FEATURES if feature != 'Pseudo_Transaction_ID'):
                return {"Pseudo_Transaction_ID": transaction_id, "Error": "Missing required data fields"}
            
            score = fraud_model_instance.predict_score(transaction_dict)
            fraud_logger.info(f"Predicted score {score:.4f} for transaction {transaction_id}")
            
            return {
                "Pseudo_Transaction_ID": transaction_id,
                "Fraud_Score": score,
                "Flagged": score >= fraud_model_instance.FRAUD_THRESHOLD
            }
        except (json.JSONDecodeError, TypeError) as e:
            fraud_logger.error(f"Error parsing transaction data: {e}")
            return {"Error": f"Invalid JSON format: {str(e)}"}
    
    @fraud_inference_mcp.tool()
    def explain_fraud_prediction(transaction_data: str) -> Optional[str]:
        """MCP tool to explain fraud prediction. Accepts JSON string."""
        import json
        try:
            if isinstance(transaction_data, str):
                transaction_dict = json.loads(transaction_data)
            else:
                transaction_dict = transaction_data
            
            score = fraud_model_instance.predict_score(transaction_dict)
            
            if score < fraud_model_instance.FRAUD_THRESHOLD:
                return f"Transaction is low risk (Score: {score:.4f})."
            
            feature_impacts = fraud_model_instance.get_feature_importance(transaction_dict, score)
            formatted_features = []
            for f in feature_impacts:
                formatted_features.append(f"{f['feature']} ({f['impact']})")
            features_summary = ", ".join(formatted_features)
            
            explanation = (
                f"🚨 **High Fraud Alert** (Score: {score:.4f}). Account: {transaction_dict.get('nameOrig')}. "
                f"The primary factors contributing to this score are: {features_summary}. " 
                "**Action:** Immediate freeze and human review recommended."
            )
            return explanation
        except (json.JSONDecodeError, TypeError) as e:
            fraud_logger.error(f"Error parsing transaction data: {e}")
            return f"Error: Invalid JSON format - {str(e)}"

# =================================================================
# ## 3. Main Execution
# =================================================================

def main(csv_file_path: Optional[str] = None, max_rows: Optional[int] = None):
    """
    Simulates the multi-agent pipeline workflow using the two MCP servers.
    The fix for the TypeError is applied here by retrieving the tool functions once.
    
    Args:
        csv_file_path: Optional path to local CSV file. If provided, uses this instead of Kaggle.
        max_rows: Optional limit on number of rows to process. If None, processes all rows.
    """
    print("\n--- Starting MCP Server Demonstration ---\n")
    
    # 1. Simulate Data Ingestion Agent calling the Data Source MCP Server
    print("DEMO: Initializing Data Source MCP Server (Ingestion Phase)...")
    
    if csv_file_path:
        print(f"Loading data from local CSV file: {csv_file_path}")
    else:
        print("Loading data from Kaggle dataset...")
    
    if max_rows:
        print(f"Processing up to {max_rows} rows...")
    else:
        print("Processing ALL rows from the CSV...")
    
    # Use external MCP client if configured, otherwise use local
    if USE_EXTERNAL_DATA_SOURCE and external_data_source_client:
        print("Loading data using external Data Source MCP server...")
        raw_data_output = external_data_source_client.fetch_raw_banking_data(
            csv_file_path=csv_file_path,
            max_rows=max_rows
        )
    else:
        # Use direct DataSourceMCP call (local fallback)
        print("Loading data using local DataSourceMCP...")
        ds_server = DataSourceMCP(csv_file_path=csv_file_path, max_rows=max_rows)
        raw_data_output = ds_server.download_and_load_data() 
    
    if raw_data_output and "Error" in raw_data_output[0]:
        error_msg = raw_data_output[0]['Error']
        print(f"\nFATAL ERROR: Could not fetch data.")
        print(f"Details: {error_msg}")
        if "File not found" in error_msg:
            print("\nTroubleshooting:")
            print("1. Make sure you provide the full path to your CSV file")
            print("2. Example: python multi_agent_banking_pipeline_with_api.py --csv \"C:\\path\\to\\your\\file.csv\"")
            print("3. Or use a relative path from the current directory")
        elif "Memory error" in error_msg or "Unable to allocate" in error_msg:
            print("\nTroubleshooting:")
            print("The dataset is too large to load entirely into memory.")
            print("Recommended solutions:")
            print("1. Use --max-rows to limit processing:")
            print("   python multi_agent_banking_pipeline_with_api.py --max-rows 100000")
            print("2. For testing, start with a smaller sample:")
            print("   python multi_agent_banking_pipeline_with_api.py --max-rows 10000")
            print("3. For full dataset processing, consider processing in batches")
        elif "Kagglehub" in error_msg:
            print("Please ensure you have run 'pip install kagglehub' and configured your Kaggle credentials.")
        return

    # 2. Simulate Data Processing Agent's action (Creating Pseudo_Transaction_ID and cleaning)
    processed_data = []
    for i, record in enumerate(raw_data_output):
        # Data Processing Step: Create a unique ID for pipeline tracking
        record['Pseudo_Transaction_ID'] = f"TID-{record.get('step', '0')}-{i}"
        # Data Processing Step: Convert necessary fields to float/clean formats (simulated by .fillna(0) earlier)
        processed_data.append(record)
        
    print(f"\nDATA PROCESSING AGENT: Processed {len(processed_data)} records and added IDs.")

    # 3. Simulate Fraud Inference Agent calling the Fraud Inference MCP tool
    print("\nDEMO: Initializing Fraud Inference MCP Server (Inference Phase)...")
    
    # Use external MCP client if configured, otherwise use local
    if USE_EXTERNAL_FRAUD_INFERENCE and external_fraud_inference_client:
        print("Using external Fraud Inference MCP server...")
        use_external_fraud = True
    else:
        # Use direct model calls (local fallback)
        print("Using local fraud detection model...")
        fi_server = FraudInferenceMCP()
        use_external_fraud = False
    
    fraud_report = []
    for transaction in processed_data:
        if use_external_fraud:
            # Use external MCP client
            result = external_fraud_inference_client.predict_fraud_score(transaction)
            if "Error" in result:
                print(f"Error from external server: {result['Error']}")
                continue
        else:
            # Use direct model calls
            score = fi_server.fraud_model.predict_score(transaction)
            transaction_id = transaction.get('Pseudo_Transaction_ID', 'N/A')
            result = {
                "Pseudo_Transaction_ID": transaction_id,
                "Fraud_Score": score,
                "Flagged": score >= fi_server.fraud_model.FRAUD_THRESHOLD
            }
        
        if result.get('Flagged', False):
            # Generate explanation
            if use_external_fraud:
                # Get explanation from external server
                explanation = external_fraud_inference_client.explain_fraud_prediction(transaction)
            else:
                # Generate explanation using direct model calls
                score = result.get('Fraud_Score', 0)
                feature_impacts = fi_server.fraud_model.get_feature_importance(transaction, score)
                formatted_features = []
                for f in feature_impacts:
                    formatted_features.append(f"{f['feature']} ({f['impact']})")
                features_summary = ", ".join(formatted_features)
                
                explanation = (
                    f"🚨 **High Fraud Alert** (Score: {score:.4f}). Account: {transaction.get('nameOrig')}. "
                    f"The primary factors contributing to this score are: {features_summary}. " 
                    "**Action:** Immediate freeze and human review recommended."
                )
            
            report_entry = {
                "ID": result["Pseudo_Transaction_ID"],
                "Score": result["Fraud_Score"],
                "Explanation": explanation,
                "Amount": transaction['amount']
            }
            fraud_report.append(report_entry)

    # 4. Final Report Synthesis (Simulated Reporting Agent)
    print("\n## 📝 Fraudulent Transaction Report")
    print(f"Total Transactions Analyzed: {len(processed_data)}")
    print(f"Total Transactions Flagged: {len(fraud_report)}\n")
    
    for entry in fraud_report:
        print(f"--- Transaction {entry['ID']} (Amount: ${entry['Amount']:.2f}) ---")
        print(f"**Fraud Score:** {entry['Score']:.4f}")
        print(f"**Actionable Summary:** {entry['Explanation']}\n")


def run_data_source_mcp_server():
    """Run the Data Source MCP server as a standalone service."""
    if data_source_mcp:
        print("Starting Data Source MCP Server...", file=sys.stderr)
        data_source_mcp.run(transport='stdio')
    else:
        print("ERROR: MCP not available. Install with: pip install fastmcp", file=sys.stderr)
        sys.exit(1)

def run_fraud_inference_mcp_server():
    """Run the Fraud Inference MCP server as a standalone service."""
    if fraud_inference_mcp:
        print("Starting Fraud Inference MCP Server...", file=sys.stderr)
        fraud_inference_mcp.run(transport='stdio')
    else:
        print("ERROR: MCP not available. Install with: pip install fastmcp", file=sys.stderr)
        sys.exit(1)

# =================================================================
# ## 4. FastAPI Web API Server (Live HTTP API with API Key Auth)
# =================================================================

if FASTAPI_AVAILABLE:
    # Initialize FastAPI app
    app = FastAPI(
        title="Fraud Detection API",
        description="Live HTTP API for fraud detection in banking transactions",
        version="1.0.0"
    )
    
    # API Key Security
    security = HTTPBearer()
    
    # Get API key from environment variable or use default for development
    # For production, set FRAUD_DETECTION_API_KEY environment variable
    DEFAULT_API_KEY = os.getenv("FRAUD_DETECTION_API_KEY", secrets.token_urlsafe(32))
    
    if DEFAULT_API_KEY == secrets.token_urlsafe(32) and not os.getenv("FRAUD_DETECTION_API_KEY"):
        print(f"WARNING: Using randomly generated API key: {DEFAULT_API_KEY}")
        print("Set FRAUD_DETECTION_API_KEY environment variable for production use.")
    
    def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
        """Verify API key from Bearer token."""
        if credentials.credentials != DEFAULT_API_KEY:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return credentials.credentials
    
    # Pydantic Models for Request/Response
    class TransactionRequest(BaseModel):
        """Transaction data model for API requests."""
        Pseudo_Transaction_ID: Optional[str] = Field(None, description="Transaction ID")
        type: str = Field(..., description="Transaction type")
        amount: float = Field(..., description="Transaction amount")
        oldbalanceOrg: float = Field(..., description="Old balance origin")
        newbalanceOrig: float = Field(..., description="New balance origin")
        oldbalanceDest: float = Field(..., description="Old balance destination")
        newbalanceDest: float = Field(..., description="New balance destination")
        nameOrig: str = Field(..., description="Origin account name")
        
        class Config:
            json_schema_extra = {
                "example": {
                    "Pseudo_Transaction_ID": "TID-1-0",
                    "type": "TRANSFER",
                    "amount": 100000.0,
                    "oldbalanceOrg": 500000.0,
                    "newbalanceOrig": 400000.0,
                    "oldbalanceDest": 0.0,
                    "newbalanceDest": 100000.0,
                    "nameOrig": "C1234567890"
                }
            }
    
    class FraudScoreResponse(BaseModel):
        """Fraud score prediction response."""
        Pseudo_Transaction_ID: str
        Fraud_Score: float
        Flagged: bool
        timestamp: str
    
    class FraudExplanationResponse(BaseModel):
        """Fraud explanation response."""
        Pseudo_Transaction_ID: str
        Fraud_Score: float
        Flagged: bool
        Explanation: str
        timestamp: str
    
    class BatchTransactionRequest(BaseModel):
        """Batch transaction processing request."""
        transactions: List[TransactionRequest] = Field(..., description="List of transactions")
        max_rows: Optional[int] = Field(None, description="Maximum rows to process")
    
    class BatchFraudResponse(BaseModel):
        """Batch fraud detection response."""
        total_transactions: int
        flagged_transactions: int
        results: List[Dict[str, Any]]
        timestamp: str
    
    # API Endpoints
    @app.get("/")
    async def root():
        """API root endpoint."""
        return {
            "message": "Fraud Detection API",
            "version": "1.0.0",
            "endpoints": {
                "predict": "/api/v1/predict",
                "explain": "/api/v1/explain",
                "batch": "/api/v1/batch",
                "health": "/health"
            },
            "documentation": "/docs"
        }
    
    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "model_threshold": fraud_model_instance.FRAUD_THRESHOLD
        }
    
    @app.post("/api/v1/predict", response_model=FraudScoreResponse)
    async def predict_fraud_score_api(
        transaction: TransactionRequest,
        api_key: str = Depends(verify_api_key)
    ):
        """Predict fraud score for a single transaction."""
        try:
            transaction_dict = transaction.model_dump()
            
            # Ensure Pseudo_Transaction_ID exists
            if not transaction_dict.get('Pseudo_Transaction_ID'):
                transaction_dict['Pseudo_Transaction_ID'] = f"TID-{datetime.now().timestamp()}"
            
            score = fraud_model_instance.predict_score(transaction_dict)
            flagged = score >= fraud_model_instance.FRAUD_THRESHOLD
            
            return FraudScoreResponse(
                Pseudo_Transaction_ID=transaction_dict['Pseudo_Transaction_ID'],
                Fraud_Score=round(score, 4),
                Flagged=flagged,
                timestamp=datetime.now().isoformat()
            )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error processing transaction: {str(e)}"
            )
    
    @app.post("/api/v1/explain", response_model=FraudExplanationResponse)
    async def explain_fraud_prediction_api(
        transaction: TransactionRequest,
        api_key: str = Depends(verify_api_key)
    ):
        """Explain fraud prediction with feature importance."""
        try:
            transaction_dict = transaction.model_dump()
            
            # Ensure Pseudo_Transaction_ID exists
            if not transaction_dict.get('Pseudo_Transaction_ID'):
                transaction_dict['Pseudo_Transaction_ID'] = f"TID-{datetime.now().timestamp()}"
            
            score = fraud_model_instance.predict_score(transaction_dict)
            flagged = score >= fraud_model_instance.FRAUD_THRESHOLD
            
            if score < fraud_model_instance.FRAUD_THRESHOLD:
                explanation = f"Transaction is low risk (Score: {score:.4f})."
            else:
                feature_impacts = fraud_model_instance.get_feature_importance(transaction_dict, score)
                formatted_features = []
                for f in feature_impacts:
                    formatted_features.append(f"{f['feature']} ({f['impact']})")
                features_summary = ", ".join(formatted_features)
                
                explanation = (
                    f"🚨 **High Fraud Alert** (Score: {score:.4f}). Account: {transaction_dict.get('nameOrig')}. "
                    f"The primary factors contributing to this score are: {features_summary}. " 
                    "**Action:** Immediate freeze and human review recommended."
                )
            
            return FraudExplanationResponse(
                Pseudo_Transaction_ID=transaction_dict['Pseudo_Transaction_ID'],
                Fraud_Score=round(score, 4),
                Flagged=flagged,
                Explanation=explanation,
                timestamp=datetime.now().isoformat()
            )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error explaining prediction: {str(e)}"
            )
    
    @app.post("/api/v1/batch", response_model=BatchFraudResponse)
    async def batch_fraud_detection_api(
        batch_request: BatchTransactionRequest,
        api_key: str = Depends(verify_api_key)
    ):
        """Process multiple transactions in batch."""
        try:
            results = []
            flagged_count = 0
            
            # Apply max_rows limit if specified
            transactions = batch_request.transactions
            if batch_request.max_rows:
                transactions = transactions[:batch_request.max_rows]
            
            for transaction in transactions:
                transaction_dict = transaction.model_dump()
                
                # Ensure Pseudo_Transaction_ID exists
                if not transaction_dict.get('Pseudo_Transaction_ID'):
                    transaction_dict['Pseudo_Transaction_ID'] = f"TID-{datetime.now().timestamp()}-{len(results)}"
                
                score = fraud_model_instance.predict_score(transaction_dict)
                flagged = score >= fraud_model_instance.FRAUD_THRESHOLD
                
                if flagged:
                    flagged_count += 1
                    feature_impacts = fraud_model_instance.get_feature_importance(transaction_dict, score)
                    formatted_features = []
                    for f in feature_impacts:
                        formatted_features.append(f"{f['feature']} ({f['impact']})")
                    features_summary = ", ".join(formatted_features)
                    
                    explanation = (
                        f"🚨 **High Fraud Alert** (Score: {score:.4f}). Account: {transaction_dict.get('nameOrig')}. "
                        f"The primary factors contributing to this score are: {features_summary}."
                    )
                else:
                    explanation = f"Transaction is low risk (Score: {score:.4f})."
                
                results.append({
                    "Pseudo_Transaction_ID": transaction_dict['Pseudo_Transaction_ID'],
                    "Fraud_Score": round(score, 4),
                    "Flagged": flagged,
                    "Explanation": explanation
                })
            
            return BatchFraudResponse(
                total_transactions=len(results),
                flagged_transactions=flagged_count,
                results=results,
                timestamp=datetime.now().isoformat()
            )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error processing batch: {str(e)}"
            )


def run_web_api_server(host: str = "0.0.0.0", port: int = 8000):
    """Run the FastAPI web API server."""
    if not FASTAPI_AVAILABLE:
        print("ERROR: FastAPI not available. Install with: pip install fastapi uvicorn", file=sys.stderr)
        sys.exit(1)
    
    # Get API key for display (same logic as in FASTAPI_AVAILABLE block)
    api_key = os.getenv("FRAUD_DETECTION_API_KEY", secrets.token_urlsafe(32))
    
    print(f"Starting Fraud Detection Web API Server...", file=sys.stderr)
    print(f"API will be available at: http://{host}:{port}", file=sys.stderr)
    print(f"API Documentation: http://{host}:{port}/docs", file=sys.stderr)
    print(f"API Key (set FRAUD_DETECTION_API_KEY env var to change): {api_key}", file=sys.stderr)
    print(f"Use this header in requests: Authorization: Bearer {api_key}", file=sys.stderr)
    
    uvicorn.run(app, host=host, port=port, log_level="info")

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Multi-Agent Banking Pipeline with Fraud Detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run demo with Kaggle dataset (first 100 rows)
  python multi_agent_banking_pipeline_with_api.py
  
  # Test whole CSV from local file
  python multi_agent_banking_pipeline_with_api.py --csv path/to/file.csv
  
  # Test first 1000 rows from local CSV
  python multi_agent_banking_pipeline_with_api.py --csv path/to/file.csv --max-rows 1000
  
  # Test whole CSV from Kaggle (all rows)
  python multi_agent_banking_pipeline_with_api.py --max-rows 0
  
  # Run MCP servers
  python multi_agent_banking_pipeline_with_api.py data-source
  python multi_agent_banking_pipeline_with_api.py fraud-inference
  
  # Run Web API server (Live HTTP API with API key auth)
  python multi_agent_banking_pipeline_with_api.py web-api
  python multi_agent_banking_pipeline_with_api.py web-api --api-host 127.0.0.1 --api-port 8000
        """
    )
    parser.add_argument('--csv', type=str, help='Path to local CSV file to process')
    parser.add_argument('--max-rows', type=int, default=None, 
                       help='Maximum number of rows to process (default: all rows, use 0 for all from Kaggle)')
    parser.add_argument('--api-host', type=str, default='0.0.0.0',
                       help='Host for web API server (default: 0.0.0.0)')
    parser.add_argument('--api-port', type=int, default=8000,
                       help='Port for web API server (default: 8000)')
    parser.add_argument('server', nargs='?', choices=['data-source', 'fraud-inference', 'web-api'],
                       help='Run as MCP server or web API server (optional)')
    
    args = parser.parse_args()
    
    # Check if running as MCP server or web API server
    if args.server:
        if args.server == "data-source":
            run_data_source_mcp_server()
        elif args.server == "fraud-inference":
            run_fraud_inference_mcp_server()
        elif args.server == "web-api":
            run_web_api_server(host=args.api_host, port=args.api_port)
        sys.exit(0)
    
    # Handle max_rows: 0 means process all rows
    max_rows = None if args.max_rows == 0 else args.max_rows
    
    # Run the demo
    main(csv_file_path=args.csv, max_rows=max_rows)
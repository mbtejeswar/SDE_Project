"""Fraud Inference MCP server implementation."""

import sys
import json
import logging
from typing import Dict, Any, Optional

from ..config import get_settings
from ..models import FraudDetectorModel

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


# Global fraud model instance for MCP tools
fraud_model_instance = FraudDetectorModel()
fraud_logger = logging.getLogger("FraudInferenceServer")


def create_fraud_inference_mcp_server():
    """Create and configure the Fraud Inference MCP server."""
    settings = get_settings()
    
    if settings.use_external_fraud_inference:
        return None
    
    if not MCP_SERVER_AVAILABLE:
        return None
    
    if USE_FASTMCP:
        fraud_inference_mcp = FastMCP("FraudInference")
        
        @fraud_inference_mcp.tool()
        def predict_score(transaction: str) -> float:
            """MCP tool to predict fraud score for a transaction. Accepts JSON string."""
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
            try:
                if isinstance(transaction_data, str):
                    transaction_dict = json.loads(transaction_data)
                else:
                    transaction_dict = transaction_data
                
                transaction_id = transaction_dict.get('Pseudo_Transaction_ID', 'N/A')
                
                if not all(feature in transaction_dict for feature in settings.EXPECTED_FEATURES if feature != 'Pseudo_Transaction_ID'):
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
        
        return fraud_inference_mcp
    
    return None


def run_fraud_inference_mcp_server():
    """Run the Fraud Inference MCP server as a standalone service."""
    fraud_inference_mcp = create_fraud_inference_mcp_server()
    
    if fraud_inference_mcp:
        print("Starting Fraud Inference MCP Server...", file=sys.stderr)
        fraud_inference_mcp.run(transport='stdio')
    else:
        print("ERROR: MCP not available. Install with: pip install fastmcp", file=sys.stderr)
        sys.exit(1)




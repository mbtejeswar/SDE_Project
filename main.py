"""Main entry point for the multi-agent banking pipeline."""

import argparse
import sys

from src.utils.logging_config import setup_logging
from src.agents import DataIngestionAgent, DataProcessingAgent, FraudInferenceAgent
from src.mcp.data_source_server import run_data_source_mcp_server
from src.mcp.fraud_inference_server import run_fraud_inference_mcp_server

# Optional API import - only needed for web-api server mode
try:
    from src.api.main import run_web_api_server
    API_AVAILABLE = True
except ImportError:
    API_AVAILABLE = False
    def run_web_api_server(*args, **kwargs):
        print("ERROR: FastAPI not available. Install with: pip install fastapi uvicorn pydantic", file=sys.stderr)
        sys.exit(1)


def main(csv_file_path: str = None, max_rows: int = None):
    """
    Simulates the multi-agent pipeline workflow using the agents.
    
    Args:
        csv_file_path: Optional path to local CSV file. If provided, uses this instead of Kaggle.
        max_rows: Optional limit on number of rows to process. If None, processes all rows.
    """
    setup_logging()
    
    print("\n--- Starting MCP Server Demonstration ---\n")
    
    # 1. Data Ingestion Agent
    print("DEMO: Initializing Data Ingestion Agent...")
    if csv_file_path:
        print(f"Loading data from local CSV file: {csv_file_path}")
    else:
        print("Loading data from Kaggle dataset...")
    
    if max_rows:
        print(f"Processing up to {max_rows} rows...")
    else:
        print("Processing ALL rows from the CSV...")
    
    ingestion_agent = DataIngestionAgent()
    raw_data_output = ingestion_agent.fetch_raw_data(
        csv_file_path=csv_file_path,
        max_rows=max_rows
    )
    
    if raw_data_output and "Error" in raw_data_output[0]:
        error_msg = raw_data_output[0]['Error']
        print(f"\nFATAL ERROR: Could not fetch data.")
        print(f"Details: {error_msg}")
        if "File not found" in error_msg:
            print("\nTroubleshooting:")
            print("1. Make sure you provide the full path to your CSV file")
            print("2. Example: python main.py --csv \"/path/to/your/file.csv\"")
            print("3. Or use a relative path from the current directory")
        elif "Memory error" in error_msg or "Unable to allocate" in error_msg:
            print("\nTroubleshooting:")
            print("The dataset is too large to load entirely into memory.")
            print("Recommended solutions:")
            print("1. Use --max-rows to limit processing:")
            print("   python main.py --max-rows 100000")
            print("2. For testing, start with a smaller sample:")
            print("   python main.py --max-rows 10000")
            print("3. For full dataset processing, consider processing in batches")
        elif "Kagglehub" in error_msg:
            print("Please ensure you have run 'pip install kagglehub' and configured your Kaggle credentials.")
        return
    
    # 2. Data Processing Agent
    print("\nDEMO: Initializing Data Processing Agent...")
    processing_agent = DataProcessingAgent()
    processed_data = processing_agent.process_transactions(raw_data_output)
    print(f"DATA PROCESSING AGENT: Processed {len(processed_data)} records and added IDs.")
    
    # 3. Fraud Inference Agent
    print("\nDEMO: Initializing Fraud Inference Agent...")
    fraud_agent = FraudInferenceAgent()
    fraud_report = fraud_agent.analyze_transactions(processed_data)
    
    # 4. Final Report Synthesis
    print("\n## 📝 Fraudulent Transaction Report")
    print(f"Total Transactions Analyzed: {len(processed_data)}")
    print(f"Total Transactions Flagged: {len(fraud_report)}\n")
    
    for entry in fraud_report:
        print(f"--- Transaction {entry['ID']} (Amount: ${entry['Amount']:.2f}) ---")
        print(f"**Fraud Score:** {entry['Score']:.4f}")
        print(f"**Actionable Summary:** {entry['Explanation']}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Multi-Agent Banking Pipeline with Fraud Detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run demo with Kaggle dataset (first 100 rows)
  python main.py
  
  # Test whole CSV from local file
  python main.py --csv path/to/file.csv
  
  # Test first 1000 rows from local CSV
  python main.py --csv path/to/file.csv --max-rows 1000
  
  # Test whole CSV from Kaggle (all rows)
  python main.py --max-rows 0
  
  # Run MCP servers
  python main.py data-source
  python main.py fraud-inference
  
  # Run Web API server (Live HTTP API with API key auth)
  python main.py web-api
  python main.py web-api --api-host 127.0.0.1 --api-port 8000
        """
    )
    parser.add_argument('--csv', type=str, help='Path to local CSV file to process')
    parser.add_argument('--max-rows', type=int, default=None, 
                       help='Maximum number of rows to process (default: all rows, use 0 for all from Kaggle)')
    parser.add_argument('--api-host', type=str, default=None,
                       help='Host for web API server (default: from env or 0.0.0.0)')
    parser.add_argument('--api-port', type=int, default=None,
                       help='Port for web API server (default: from env or 8000)')
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
            if not API_AVAILABLE:
                print("ERROR: FastAPI dependencies not installed.", file=sys.stderr)
                print("Install with: pip install fastapi uvicorn pydantic", file=sys.stderr)
                sys.exit(1)
            run_web_api_server(host=args.api_host, port=args.api_port)
        sys.exit(0)
    
    # Handle max_rows: 0 means process all rows
    max_rows = None if args.max_rows == 0 else args.max_rows
    
    # Run the demo
    main(csv_file_path=args.csv, max_rows=max_rows)




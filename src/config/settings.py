"""Application settings and configuration."""

import os
from typing import Optional


class Settings:
    """Application settings."""
    
    EXPECTED_FEATURES = [
        'Pseudo_Transaction_ID', 'type', 'amount', 'oldbalanceOrg', 
        'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'nameOrig'
    ]

    FRAUD_THRESHOLD = 0.50
    
    DEFAULT_DATASET_ID = 'rupakroy/online-payments-fraud-detection-dataset'

    DEFAULT_CSV_FILENAME = 'PS_20174392719_1491204439457_log.csv'

    EXTERNAL_DATA_SOURCE_MCP_URL: Optional[str] = os.getenv("EXTERNAL_DATA_SOURCE_MCP_URL", None)
    EXTERNAL_FRAUD_INFERENCE_MCP_URL: Optional[str] = os.getenv("EXTERNAL_FRAUD_INFERENCE_MCP_URL", None)
    

    FRAUD_DETECTION_API_KEY: Optional[str] = os.getenv("FRAUD_DETECTION_API_KEY", None)
    
    # API server settings
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    
    @property
    def use_external_data_source(self) -> bool:
        """Check if external data source MCP server should be used."""
        return self.EXTERNAL_DATA_SOURCE_MCP_URL is not None
    
    @property
    def use_external_fraud_inference(self) -> bool:
        """Check if external fraud inference MCP server should be used."""
        return self.EXTERNAL_FRAUD_INFERENCE_MCP_URL is not None


# Global settings instance
_settings = None


def get_settings() -> Settings:
    """Get or create settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings




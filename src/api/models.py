"""Pydantic models for API requests and responses."""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


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




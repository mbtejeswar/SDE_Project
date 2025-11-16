"""FastAPI application main module."""

import sys
from datetime import datetime
from typing import List, Dict, Any

# Try to import FastAPI
FASTAPI_AVAILABLE = False
try:
    from fastapi import FastAPI, HTTPException, Depends
    from fastapi.responses import JSONResponse
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    pass

from ..config import get_settings
from ..models import FraudDetectorModel
from .models import (
    TransactionRequest,
    FraudScoreResponse,
    FraudExplanationResponse,
    BatchTransactionRequest,
    BatchFraudResponse
)
from .dependencies import verify_api_key


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    if not FASTAPI_AVAILABLE:
        raise RuntimeError("FastAPI not available. Install with: pip install fastapi uvicorn")
    
    app = FastAPI(
        title="Fraud Detection API",
        description="Live HTTP API for fraud detection in banking transactions",
        version="1.0.0"
    )
    
    settings = get_settings()
    fraud_model = FraudDetectorModel()
    
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
            "model_threshold": fraud_model.FRAUD_THRESHOLD
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
            
            score = fraud_model.predict_score(transaction_dict)
            flagged = score >= fraud_model.FRAUD_THRESHOLD
            
            return FraudScoreResponse(
                Pseudo_Transaction_ID=transaction_dict['Pseudo_Transaction_ID'],
                Fraud_Score=round(score, 4),
                Flagged=flagged,
                timestamp=datetime.now().isoformat()
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
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
            
            score = fraud_model.predict_score(transaction_dict)
            flagged = score >= fraud_model.FRAUD_THRESHOLD
            
            if score < fraud_model.FRAUD_THRESHOLD:
                explanation = f"Transaction is low risk (Score: {score:.4f})."
            else:
                feature_impacts = fraud_model.get_feature_importance(transaction_dict, score)
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
                status_code=500,
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
                
                score = fraud_model.predict_score(transaction_dict)
                flagged = score >= fraud_model.FRAUD_THRESHOLD
                
                if flagged:
                    flagged_count += 1
                    feature_impacts = fraud_model.get_feature_importance(transaction_dict, score)
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
                status_code=500,
                detail=f"Error processing batch: {str(e)}"
            )
    
    return app


# Create app instance
if FASTAPI_AVAILABLE:
    app = create_app()
else:
    app = None


def run_web_api_server(host: str = None, port: int = None):
    """Run the FastAPI web API server."""
    if not FASTAPI_AVAILABLE:
        print("ERROR: FastAPI not available. Install with: pip install fastapi uvicorn", file=sys.stderr)
        sys.exit(1)
    
    settings = get_settings()
    host = host or settings.API_HOST
    port = port or settings.API_PORT
    
    from .dependencies import get_api_key
    api_key = get_api_key()
    
    print(f"Starting Fraud Detection Web API Server...", file=sys.stderr)
    print(f"API will be available at: http://{host}:{port}", file=sys.stderr)
    print(f"API Documentation: http://{host}:{port}/docs", file=sys.stderr)
    print(f"API Key (set FRAUD_DETECTION_API_KEY env var to change): {api_key}", file=sys.stderr)
    print(f"Use this header in requests: Authorization: Bearer {api_key}", file=sys.stderr)
    
    uvicorn.run(app, host=host, port=port, log_level="info")




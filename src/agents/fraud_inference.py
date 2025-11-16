"""Fraud Inference Agent."""

import logging
from typing import List, Dict, Any, Optional

from ..models import FraudDetectorModel
from ..mcp.clients import ExternalFraudInferenceMCPClient
from ..config import get_settings


class FraudInferenceAgent:
    
    def __init__(self):
        self.logger = logging.getLogger("FraudInferenceAgent")
        self.settings = get_settings()
        self.fraud_model = FraudDetectorModel()
        self.external_client = None
        
        if self.settings.use_external_fraud_inference:
            try:
                self.external_client = ExternalFraudInferenceMCPClient(
                    self.settings.EXTERNAL_FRAUD_INFERENCE_MCP_URL
                )
            except Exception as e:
                self.logger.warning(f"Failed to create external Fraud: {e}")
    
    def predict_fraud_score(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        transaction_id = transaction.get('Pseudo_Transaction_ID', 'N/A')
        
        if self.settings.use_external_fraud_inference and self.external_client:
            result = self.external_client.predict_fraud_score(transaction)
            if "Error" in result:
                self.logger.error(f"Error from external server: {result['Error']}")
                return result
            return result
        else:
            score = self.fraud_model.predict_score(transaction)
            return {
                "Pseudo_Transaction_ID": transaction_id,
                "Fraud_Score": score,
                "Flagged": score >= self.fraud_model.FRAUD_THRESHOLD
            }
    
    def explain_fraud_prediction(self, transaction: Dict[str, Any], score: Optional[float] = None) -> str:

        if score is None:
            result = self.predict_fraud_score(transaction)
            score = result.get('Fraud_Score', 0)
        
        if self.settings.use_external_fraud_inference and self.external_client:
            explanation = self.external_client.explain_fraud_prediction(transaction)
            return explanation or f"Transaction is low risk (Score: {score:.4f})."
        else:
            if score < self.fraud_model.FRAUD_THRESHOLD:
                return f"Transaction is low risk (Score: {score:.4f})."
            
            feature_impacts = self.fraud_model.get_feature_importance(transaction, score)
            formatted_features = []
            for f in feature_impacts:
                formatted_features.append(f"{f['feature']} ({f['impact']})")
            features_summary = ", ".join(formatted_features)
            
            explanation = (
                f"🚨 **High Fraud Alert** (Score: {score:.4f}). Account: {transaction.get('nameOrig')}. "
                f"The primary factors contributing to this score are: {features_summary}. " 
                "**Action:** Immediate freeze and human review recommended."
            )
            return explanation
    
    def analyze_transactions(self, transactions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        fraud_report = []
        
        for transaction in transactions:
            result = self.predict_fraud_score(transaction)
            
            if result.get('Flagged', False):
                explanation = self.explain_fraud_prediction(
                    transaction, 
                    result.get('Fraud_Score')
                )
                
                report_entry = {
                    "ID": result["Pseudo_Transaction_ID"],
                    "Score": result["Fraud_Score"],
                    "Explanation": explanation,
                    "Amount": transaction.get('amount', 0)
                }
                fraud_report.append(report_entry)
        
        return fraud_report




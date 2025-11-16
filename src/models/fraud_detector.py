"""Fraud detection model."""

from typing import List, Dict, Any
from ..config import get_settings


class FraudDetectorModel:
    """Simulated ML Model for fraud detection."""
    
    def __init__(self):
        """Initialize fraud detector with threshold from settings."""
        settings = get_settings()
        self.FRAUD_THRESHOLD = settings.FRAUD_THRESHOLD

    def predict_score(self, transaction: Dict[str, Any]) -> float:
        """Predict fraud score for a transaction.
        
        Args:
            transaction: Transaction data dictionary
            
        Returns:
            Fraud score between 0.0 and 1.0
        """
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
        """Get feature importance for a transaction.
        
        Args:
            transaction: Transaction data dictionary
            score: Predicted fraud score
            
        Returns:
            List of feature impact dictionaries
        """
        if score < self.FRAUD_THRESHOLD:
            return []
        
        impacts = []
        if transaction.get('type') in ['TRANSFER', 'CASH_OUT']:
            impacts.append({"feature": "type", "impact": "High (Common fraud vector)"})
        if transaction.get('amount', 0) > 100000:
            impacts.append({"feature": "amount", "impact": "High (Value is unusually large)"})
        if transaction.get('oldbalanceOrg', 0) > 0.0 and transaction.get('newbalanceOrig', 0) == 0.0:
            impacts.append({"feature": "balance depletion", "impact": "Critical (Originator's account was drained)"})
        return impacts




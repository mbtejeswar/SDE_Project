"""Data Processing Agent."""

import logging
from typing import List, Dict, Any


class DataProcessingAgent:
    """Agent responsible for processing and cleaning transaction data."""
    
    def __init__(self):
        self.logger = logging.getLogger("DataProcessingAgent")
    
    def process_transactions(self, raw_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process raw transaction data and add tracking IDs.
        
        Args:
            raw_data: List of raw transaction records
            
        Returns:
            List of processed transaction records with IDs
        """
        processed_data = []
        for i, record in enumerate(raw_data):
            # Create a unique ID for pipeline tracking
            record['Pseudo_Transaction_ID'] = f"TID-{record.get('step', '0')}-{i}"
            processed_data.append(record)
        
        self.logger.info(f"Processed {len(processed_data)} records and added IDs.")
        return processed_data




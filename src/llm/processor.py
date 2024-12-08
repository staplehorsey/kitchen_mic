#!/usr/bin/env python3
"""Process transcribed conversations using LLM."""

import json
import logging
from pathlib import Path
from typing import Dict, Optional
import time

from ..messages import TranscriptionMessage, SummaryMessage
from .client import LLMClient

logger = logging.getLogger(__name__)

class SummaryProcessor:
    """Process transcribed conversations to generate summaries."""
    
    def __init__(self, llm_client: Optional[LLMClient] = None):
        """Initialize summary processor.
        
        Args:
            llm_client: LLM client instance, creates new one if None
        """
        self.llm_client = llm_client or LLMClient()
        
    def process_conversation(self, message: TranscriptionMessage) -> Optional[SummaryMessage]:
        """Process conversation transcription to generate summary.
        
        Args:
            message: TranscriptionMessage containing conversation data
            
        Returns:
            SummaryMessage with results if successful, None otherwise
        """
        # Check transcription data
        logger.info(f"Processing message with transcription: {message.transcription}")
        if not message.transcription or "text" not in message.transcription:
            logger.warning("No transcription.text field in message")
            return None
            
        # Generate summary
        start_time = time.time()
        result = self.llm_client.generate_summary(message.transcription["text"])
        if not result:
            return None
            
        # Add timing metadata
        result["timing"] = {
            "process_start": start_time,
            "process_end": time.time()
        }
        
        # Create summary message
        return SummaryMessage.from_transcription(message, summary=result)
        
    def save_summary(self, message: SummaryMessage, output_path: Path) -> None:
        """Save summary to JSON file.
        
        Args:
            message: Summary message to save
            output_path: Path to save JSON file
        """
        with open(output_path, "w") as f:
            json.dump(message.to_dict(), f, indent=2)

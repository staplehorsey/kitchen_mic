#!/usr/bin/env python3
"""Test script for LLM summary generation.

This script tests the LLM summary system by:
1. Loading the latest transcribed conversation
2. Generating a summary using Llama
3. Saving the summary results
"""

import json
import logging
import sys
from pathlib import Path

# Add src directory to Python path
src_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(src_dir))

from src.messages import TranscriptionMessage
from src.llm.processor import SummaryProcessor

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def main():
    """Run summary generation test."""
    # Get latest transcription file
    trans_dir = Path("debug/transcriptions")
    if not trans_dir.exists():
        logger.error("No transcriptions directory found")
        return
        
    # Find transcription files
    json_files = list(trans_dir.glob("*.json"))
    if not json_files:
        logger.error("No JSON files found")
        return
        
    # Get latest transcription file
    trans_path = max(json_files, key=lambda p: p.stat().st_mtime)
    logger.info(f"Processing transcription: {trans_path}")
    
    # Load transcription data
    with open(trans_path) as f:
        data = json.load(f)
        
    logger.info(f"Loaded data keys: {list(data.keys())}")
    if "transcription" in data:
        logger.info(f"Transcription keys: {list(data['transcription'].keys())}")
    
    # Create transcription message
    message = TranscriptionMessage(
        id=data["id"],
        start_time=data["start_time"],
        end_time=data["end_time"],
        audio_data=None,  # We don't need audio for summary
        audio_data_16k=None,
        metadata=data["metadata"],
        transcription=data.get("transcription")  # Transcription is at root level
    )
    
    # Create summary processor
    processor = SummaryProcessor()
    
    # Generate summary
    summary = processor.process_conversation(message)
    if not summary:
        logger.error("Failed to generate summary")
        return
        
    # Save summary
    summary_path = trans_path.parent / f"{trans_path.stem}_summary.json"
    processor.save_summary(summary, summary_path)
    logger.info(f"Saved summary to {summary_path}")
    
    # Print summary
    print("\nGenerated Summary:")
    print(f"Title: {summary.summary['title']}")
    print(f"\nSummary: {summary.summary['summary']}")
    print("\nTopics:")
    for topic in summary.summary['topics']:
        print(f"- {topic}")

if __name__ == "__main__":
    main()

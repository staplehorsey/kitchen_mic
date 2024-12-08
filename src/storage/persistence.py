"""Handles persistent storage of conversations and metadata.

Organizes conversations by date and title, storing audio, transcriptions,
and metadata in a clean directory structure.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict
import soundfile as sf
import numpy as np

from ..messages import ConversationMessage, TranscriptionMessage, SummaryMessage

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConversationStorage:
    """Manages persistent storage of conversations."""
    
    def __init__(self, base_dir: Path):
        """Initialize storage manager.
        
        Args:
            base_dir: Base directory for all conversation storage
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Storage initialized at {self.base_dir}")
    
    def _get_conversation_dir(self, timestamp: float, title: Optional[str] = None) -> Path:
        """Get directory path for a conversation.
        
        Args:
            timestamp: Conversation timestamp
            title: Optional conversation title (from summary)
            
        Returns:
            Path to conversation directory
        """
        # Convert timestamp to datetime
        dt = datetime.fromtimestamp(timestamp)
        
        # Create year/month/day path
        date_path = self.base_dir / str(dt.year) / f"{dt.month:02d}" / f"{dt.day:02d}"
        
        # Add timestamp and title
        if title:
            # Clean title for filesystem
            clean_title = "".join(c for c in title if c.isalnum() or c in " -_").strip()
            clean_title = clean_title[:50]  # Limit length
            dir_name = f"{dt.strftime('%H%M%S')}_{clean_title}"
        else:
            dir_name = dt.strftime('%H%M%S')
            
        conv_dir = date_path / dir_name
        conv_dir.mkdir(parents=True, exist_ok=True)
        
        return conv_dir
    
    def save_conversation(
        self,
        conversation: ConversationMessage,
        transcription: Optional[TranscriptionMessage] = None,
        summary: Optional[SummaryMessage] = None
    ) -> Path:
        """Save a complete conversation with all available data.
        
        Args:
            conversation: Base conversation message
            transcription: Optional transcription results
            summary: Optional summary results
            
        Returns:
            Path to conversation directory
        """
        # Get title from summary if available
        title = None
        if summary and summary.summary:
            title = summary.summary.get('title')
        
        # Get conversation directory
        conv_dir = self._get_conversation_dir(conversation.start_time, title)
        logger.info(f"Saving conversation to {conv_dir}")
        
        try:
            # Save 44kHz audio
            audio_path = conv_dir / "audio_44k.wav"
            sf.write(
                audio_path,
                conversation.audio_data,
                samplerate=44100,
                subtype='PCM_16'
            )
            
            # Build metadata
            metadata = {
                'id': conversation.id,
                'start_time': conversation.start_time,
                'end_time': conversation.end_time,
                'duration': conversation.duration,
                'base_metadata': conversation.metadata,
            }
            
            # Add transcription if available
            if transcription and transcription.transcription:
                metadata['transcription'] = {
                    'text': transcription.transcription['text'],
                    'language': transcription.transcription.get('language'),
                    'timing': transcription.transcription.get('timing')
                }
            
            # Add summary if available
            if summary and summary.summary:
                metadata['summary'] = summary.summary
            
            # Save metadata
            metadata_path = conv_dir / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(
                f"Saved conversation {conversation.id} "
                f"({conversation.duration:.1f}s)"
            )
            return conv_dir
            
        except Exception as e:
            logger.error(
                f"Error saving conversation {conversation.id}: {e}",
                exc_info=True
            )
            raise
    
    def list_conversations(
        self,
        year: Optional[int] = None,
        month: Optional[int] = None,
        day: Optional[int] = None
    ) -> Dict:
        """List saved conversations, optionally filtered by date.
        
        Args:
            year: Optional year to filter
            month: Optional month to filter (1-12)
            day: Optional day to filter (1-31)
            
        Returns:
            Dict of conversation metadata indexed by path
        """
        # Build search path
        search_path = self.base_dir
        if year:
            search_path = search_path / str(year)
            if month:
                search_path = search_path / f"{month:02d}"
                if day:
                    search_path = search_path / f"{day:02d}"
        
        # Find all metadata files
        results = {}
        for meta_path in search_path.rglob("metadata.json"):
            try:
                with open(meta_path) as f:
                    metadata = json.load(f)
                results[str(meta_path.parent)] = metadata
            except Exception as e:
                logger.error(f"Error reading {meta_path}: {e}")
                continue
                
        return results

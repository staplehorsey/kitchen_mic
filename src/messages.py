"""
Message types for communication between components.
Each message type represents a stage in the conversation processing pipeline.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import numpy as np
from datetime import datetime

@dataclass
class ConversationMessage:
    """Base message type for conversation data."""
    
    # Core conversation info
    id: str
    start_time: float  # Unix timestamp
    end_time: float    # Unix timestamp
    
    # Audio data (both streams)
    audio_data: np.ndarray      # Original 44kHz
    audio_data_16k: np.ndarray  # Downsampled 16kHz for processing
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration(self) -> float:
        """Get conversation duration in seconds."""
        return self.end_time - self.start_time
    
    def add_timestamps(self, timestamps: List[float]) -> None:
        """Add chunk timestamps to metadata.
        
        Args:
            timestamps: List of Unix timestamps for each audio chunk
        """
        self.metadata["chunk_timestamps"] = timestamps
    
    def add_vad_events(self, events: List[Dict]) -> None:
        """Add VAD events to metadata.
        
        Args:
            events: List of VAD events with timing and probability
        """
        self.metadata["vad_events"] = events
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization.
        
        Note: audio_data is excluded as it should be handled separately.
        """
        return {
            "id": self.id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "metadata": self.metadata
        }

@dataclass
class TranscriptionMessage(ConversationMessage):
    """Message type that adds transcription data."""
    
    # Transcription data (from Whisper)
    transcription: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        data = super().to_dict()
        data["transcription"] = self.transcription
        return data
    
    @classmethod
    def from_conversation(
        cls, 
        conversation: ConversationMessage,
        transcription: Optional[Dict[str, Any]] = None
    ) -> "TranscriptionMessage":
        """Create from ConversationMessage.
        
        Args:
            conversation: Base conversation message
            transcription: Optional transcription data
            
        Returns:
            New TranscriptionMessage with conversation data
        """
        return cls(
            id=conversation.id,
            start_time=conversation.start_time,
            end_time=conversation.end_time,
            audio_data=conversation.audio_data,
            audio_data_16k=conversation.audio_data_16k,
            metadata=conversation.metadata.copy(),
            transcription=transcription
        )

@dataclass
class SummaryMessage(TranscriptionMessage):
    """Message type that adds LLM summary data."""
    
    # Summary data (from LLM)
    summary: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        data = super().to_dict()
        data["summary"] = self.summary
        return data
    
    @classmethod
    def from_transcription(
        cls, 
        transcription: TranscriptionMessage,
        summary: Optional[Dict[str, Any]] = None
    ) -> "SummaryMessage":
        """Create from TranscriptionMessage.
        
        Args:
            transcription: Base transcription message
            summary: Optional summary data
            
        Returns:
            New SummaryMessage with transcription data
        """
        return cls(
            id=transcription.id,
            start_time=transcription.start_time,
            end_time=transcription.end_time,
            audio_data=transcription.audio_data,
            audio_data_16k=transcription.audio_data_16k,
            metadata=transcription.metadata.copy(),
            transcription=transcription.transcription,
            summary=summary
        )

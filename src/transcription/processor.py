"""
Transcription processor using Whisper for speech-to-text conversion.
Handles conversation audio processing while maintaining timing information.
"""

import logging
import threading
import queue
from pathlib import Path
from typing import Optional, Dict, Callable
import json

import whisper
import torch

from ..messages import ConversationMessage, TranscriptionMessage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TranscriptionProcessor:
    def __init__(
        self,
        storage_dir: Optional[Path] = None,
        on_transcription: Optional[Callable[[TranscriptionMessage], None]] = None
    ):
        """Initialize the transcription processor.
        
        Args:
            storage_dir: Directory to store transcriptions
            on_transcription: Callback when transcription is complete
        """
        self.storage_dir = storage_dir or Path("data/transcriptions")
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.on_transcription = on_transcription
        
        # Initialize Whisper
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading Whisper model on {self.device}")
        self.model = whisper.load_model("base").to(self.device)
        
        # Processing queue and thread
        self.queue = queue.Queue()
        self._running = True
        self._process_thread = threading.Thread(target=self._process_loop, daemon=True)
        self._process_thread.start()
        
        logger.info("Transcription processor initialized")
    
    def process_conversation(self, conversation: ConversationMessage) -> None:
        """Queue a conversation for transcription.
        
        Args:
            conversation: Conversation to transcribe
        """
        self.queue.put(conversation)
    
    def _process_loop(self) -> None:
        """Background thread for processing conversations."""
        while self._running:
            try:
                # Get next conversation (blocking)
                conversation = self.queue.get(timeout=1.0)
                
                try:
                    # Process conversation
                    result = self._transcribe_conversation(conversation)
                    
                    # Create transcription message
                    message = TranscriptionMessage.from_conversation(
                        conversation=conversation,
                        transcription=result
                    )
                    
                    # Store results
                    self._store_transcription(message)
                    
                    # Call callback if provided
                    if self.on_transcription:
                        try:
                            self.on_transcription(message)
                        except Exception as e:
                            logger.error(f"Error in transcription callback: {e}")
                    
                except Exception as e:
                    logger.error(f"Error processing conversation: {e}")
                
                self.queue.task_done()
            
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in process loop: {e}")
    
    def _transcribe_conversation(self, conversation: ConversationMessage) -> Dict:
        """Transcribe conversation audio.
        
        Args:
            conversation: Conversation to transcribe
            
        Returns:
            Dict containing transcription and metadata
        """
        # Run Whisper inference (handles 44kHz directly)
        result = self.model.transcribe(conversation.audio_data)
        
        # Add conversation metadata
        result.update({
            "conversation_id": conversation.id,
            "start_time": conversation.start_time,
            "end_time": conversation.end_time
        })
        
        return result
    
    def _store_transcription(self, message: TranscriptionMessage) -> None:
        """Store transcription results.
        
        Args:
            message: Transcription message to store
        """
        # Create transcription directory
        trans_dir = self.storage_dir / message.id
        trans_dir.mkdir(parents=True, exist_ok=True)
        
        # Save transcription data
        trans_path = trans_dir / "transcription.json"
        with open(trans_path, "w") as f:
            json.dump(message.to_dict(), f, indent=2)
        
        logger.info(f"Stored transcription for conversation {message.id}")
    
    def stop(self) -> None:
        """Stop the processor and clean up."""
        self._running = False
        if self._process_thread.is_alive():
            self._process_thread.join(timeout=1.0)

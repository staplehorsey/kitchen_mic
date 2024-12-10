"""Orchestrates conversation processing in background.

Coordinates between ConversationDetector and processors to handle
completed conversations without blocking new audio capture.
"""

import logging
import queue
import threading
import time
from pathlib import Path
from typing import Optional

from ..messages import ConversationMessage, TranscriptionMessage, SummaryMessage
from ..transcription.processor import TranscriptionProcessor
from ..llm.processor import SummaryProcessor
from ..storage.persistence import ConversationStorage
from .detector import ConversationDetector

# Configure logging
logger = logging.getLogger(__name__)

class ConversationOrchestrator:
    """Orchestrates conversation processing in background."""
    
    def __init__(
        self,
        detector: ConversationDetector,
        storage: ConversationStorage,
        transcription_processor: Optional[TranscriptionProcessor] = None,
        summary_processor: Optional[SummaryProcessor] = None,
        max_queue_size: int = 100
    ):
        """Initialize orchestrator.
        
        Args:
            detector: ConversationDetector instance
            storage: ConversationStorage instance
            transcription_processor: Optional TranscriptionProcessor (creates if None)
            summary_processor: Optional SummaryProcessor (creates if None)
            max_queue_size: Maximum conversations to queue
        """
        self.detector = detector
        self.storage = storage
        self.max_queue_size = max_queue_size
        
        # Use provided processors or create new ones
        self.transcription = transcription_processor or TranscriptionProcessor()
        self.summary = summary_processor or SummaryProcessor()
        
        # Initialize queue and worker
        self.queue = queue.Queue(maxsize=max_queue_size)
        self.worker_thread = None
        self.running = False
        
        # Set up detector callback
        self.detector.on_conversation = self._handle_conversation
        
        logger.debug(f"Orchestrator initialized with {max_queue_size} max queue size")
    
    def _handle_conversation(self, conversation: ConversationMessage) -> None:
        """Handle completed conversation from detector.
        
        Args:
            conversation: Completed conversation
        """
        try:
            # Add to queue, with timeout to avoid blocking detector
            self.queue.put(conversation, timeout=1.0)
            logger.info(
                f"Queued conversation {conversation.id} for processing "
                f"(duration: {conversation.duration:.1f}s)"
            )
        except queue.Full:
            logger.error("Queue full, dropping conversation")
    
    def _process_conversation(self, conversation: ConversationMessage) -> None:
        """Process a single conversation through pipeline.
        
        Args:
            conversation: Conversation to process
        """
        try:
            logger.info(f"=== Processing conversation {conversation.id} ===")
            
            # Step 1: Transcription
            logger.info("Starting transcription...")
            trans_msg = self.transcription.process_conversation(conversation)
            if not trans_msg or not trans_msg.transcription:
                logger.error(f"Transcription failed for {conversation.id}")
                return

            logger.info(f"Transcription complete")
            
            # Step 2: Summary
            logger.info("Generating summary...")
            summary_msg = self.summary.process_conversation(trans_msg)
            logger.info(f"Summary complete")
            
            # Step 3: Save everything
            logger.info("Saving conversation data...")
            self.storage.save_conversation(
                conversation=conversation,
                transcription=trans_msg,
                summary=summary_msg
            )
            logger.info(f"=== Finished processing conversation {conversation.id} ===")
            
        except Exception as e:
            logger.error(f"Error processing conversation: {e}", exc_info=True)
    
    def _worker_loop(self) -> None:
        """Background worker loop to process conversations."""
        logger.debug("Background worker started")
        
        while self.running:
            try:
                # Get next conversation with timeout
                conversation = self.queue.get(timeout=1.0)
                
                # Process it
                self._process_conversation(conversation)
                
                # Mark as done
                self.queue.task_done()
                
            except queue.Empty:
                # No conversations to process
                continue
            except Exception as e:
                logger.error(f"Worker error: {e}", exc_info=True)
                # Continue running despite errors
                continue
        
        logger.debug("Background worker stopped")
    
    def start(self) -> None:
        """Start background processing."""
        if self.running:
            logger.warning("Already running")
            return
        
        # Start worker thread
        self.running = True
        self.worker_thread = threading.Thread(
            target=self._worker_loop,
            name="ConversationWorker"
        )
        self.worker_thread.daemon = True
        self.worker_thread.start()
        
        # Start detector
        self.detector.start()
        
        logger.info("Started orchestrator")
    
    def stop(self) -> None:
        """Stop background processing."""
        if not self.running:
            logger.warning("Not running")
            return
            
        # Stop accepting new conversations
        self.detector.stop()
        
        # Stop worker
        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=5.0)
        
        logger.info("Stopped orchestrator")
    
    def get_status(self) -> dict:
        """Get current processing status.
        
        Returns:
            Dict with queue size and running state
        """
        return {
            'running': self.running,
            'queue_size': self.queue.qsize(),
            'detector_running': self.detector._running
        }

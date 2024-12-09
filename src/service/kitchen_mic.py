#!/usr/bin/env python3
"""Kitchen Mic Service Runner.

This script handles starting, stopping, and managing the Kitchen Mic service.
It's designed to work with systemd and provides proper service lifecycle management.
"""

import os
import sys
import time
import signal
import logging
import argparse
from pathlib import Path
from typing import Optional
import sdnotify  # For systemd notification

from ..config import ConfigManager
from ..conversation.orchestrator import ConversationOrchestrator
from ..conversation.detector import ConversationDetector
from ..audio.capture import AudioCapture
from ..vad.processor import VADProcessor, VADConfig
from ..storage.persistence import ConversationStorage
from ..transcription.processor import TranscriptionProcessor
from ..llm.processor import SummaryProcessor

class KitchenMicService:
    """Kitchen Mic Service Manager."""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize service.
        
        Args:
            config_path: Optional path to config file.
        """
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.get_config()
        self.setup_logging()
        self.orchestrator: Optional[ConversationOrchestrator] = None
        self.running = False
        self.notifier = sdnotify.SystemdNotifier()

    def setup_logging(self) -> None:
        """Configure logging based on service settings."""
        log_config = self.config['service']
        log_file = os.path.expanduser(log_config.get('log_file', ''))
        
        handlers = []
        if log_file:
            log_dir = os.path.dirname(log_file)
            os.makedirs(log_dir, exist_ok=True)
            handlers.append(logging.FileHandler(log_file))
        
        # Always add console handler for systemd journal
        handlers.append(logging.StreamHandler())
        
        logging.basicConfig(
            level=getattr(logging, log_config['log_level']),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=handlers
        )

    def initialize_components(self) -> None:
        """Initialize all service components."""
        try:
            # Initialize storage
            storage = ConversationStorage(self.config_manager.get_storage_path())
            
            # Initialize processors
            transcription = TranscriptionProcessor(
                model_name=self.config['models']['transcription']['name'],
                device=self.config['models']['transcription'].get('device')
            )
            summary = SummaryProcessor()
            
            # Initialize audio and VAD
            vad_config = VADConfig(
                threshold=self.config['vad']['threshold'],
                conversation_threshold=self.config['vad']['threshold'],
                conversation_cooldown_sec=self.config['vad']['conversation_cooldown_sec'],
                min_conversation_duration_sec=self.config['vad']['min_conversation_sec'],
                sample_rate=self.config['audio']['vad']['sample_rate']
            )
            vad = VADProcessor(config=vad_config)
            audio = AudioCapture(
                host=self.config['audio']['source']['host'],
                port=self.config['audio']['source']['port'],
                buffer_size=self.config['audio']['source']['buffer_size'],
                original_rate=self.config['audio']['capture']['sample_rate'],
                target_rate=self.config['audio']['vad']['sample_rate'],
                channels=self.config['audio']['capture']['channels'],
                chunk_size=self.config['audio']['vad']['chunk_size'],
                visualize=self.config['audio']['source'].get('visualize', True)
            )
            
            # Initialize detector
            detector = ConversationDetector(
                audio_processor=audio,
                vad_processor=vad,
                buffer_duration_sec=self.config['conversation']['buffer_duration_sec'],
                pre_speech_sec=self.config['vad']['pre_speech_sec']
            )
            
            # Create orchestrator
            self.orchestrator = ConversationOrchestrator(
                detector=detector,
                storage=storage,
                transcription_processor=transcription,
                summary_processor=summary,
                max_queue_size=self.config['conversation']['max_queue_size']
            )
            
        except Exception as e:
            logging.error(f"Failed to initialize components: {e}")
            raise

    def signal_handler(self, signum: int, frame) -> None:
        """Handle system signals.
        
        Args:
            signum: Signal number.
            frame: Current stack frame.
        """
        logging.info(f"Received signal {signum}")
        self.stop()

    def start(self) -> None:
        """Start the Kitchen Mic service."""
        if self.running:
            logging.warning("Service is already running")
            return
            
        try:
            logging.info("Starting Kitchen Mic service")
            self.initialize_components()
            
            # Register signal handlers
            signal.signal(signal.SIGTERM, self.signal_handler)
            signal.signal(signal.SIGINT, self.signal_handler)
            
            # Start the orchestrator
            if self.orchestrator:
                self.orchestrator.start()
                self.running = True
                
                # Notify systemd we're ready
                self.notifier.notify("READY=1")
                logging.info("Kitchen Mic service started successfully")
                
                # Keep the service running
                while self.running:
                    # Send watchdog notification if configured
                    self.notifier.notify("WATCHDOG=1")
                    time.sleep(1)
                    
        except Exception as e:
            logging.error(f"Failed to start service: {e}")
            self.notifier.notify("STATUS=Failed to start service")
            raise

    def stop(self) -> None:
        """Stop the Kitchen Mic service."""
        if not self.running:
            logging.warning("Service is not running")
            return
            
        try:
            logging.info("Stopping Kitchen Mic service")
            self.running = False
            
            if self.orchestrator:
                self.orchestrator.stop()
                self.orchestrator = None
                
            self.notifier.notify("STOPPING=1")
            logging.info("Kitchen Mic service stopped successfully")
            
        except Exception as e:
            logging.error(f"Error stopping service: {e}")
            raise

    def restart(self) -> None:
        """Restart the Kitchen Mic service."""
        logging.info("Restarting Kitchen Mic service")
        self.stop()
        self.start()


def main() -> None:
    """Main entry point for the service."""
    parser = argparse.ArgumentParser(description="Kitchen Mic Service Manager")
    parser.add_argument('command', choices=['start', 'stop', 'restart', 'health'],
                       help="Command to execute")
    parser.add_argument('--config', help="Path to config file")
    args = parser.parse_args()
    
    service = KitchenMicService(args.config)
    
    try:
        if args.command == 'health':
            # Print config and check components
            print("\nCurrent Configuration:")
            print("----------------------")
            for section, values in service.config.items():
                print(f"\n[{section}]")
                for key, value in values.items():
                    print(f"{key}: {value}")
            
            print("\nComponent Status:")
            print("----------------")
            status = {
                "config": "OK" if service.config else "Not loaded",
                "audio_source": "Connected" if service.config_manager.check_audio_source() else "Not connected",
                "llm_endpoint": "Connected" if service.config_manager.check_llm_endpoint() else "Not connected",
                "storage": "OK" if os.path.exists(service.config['storage']['base_dir']) else "Not available"
            }
            for component, state in status.items():
                print(f"{component}: {state}")
            sys.exit(0)
        elif args.command == 'start':
            service.start()
        elif args.command == 'stop':
            service.stop()
        elif args.command == 'restart':
            service.restart()
    except Exception as e:
        logging.error(f"Service command '{args.command}' failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

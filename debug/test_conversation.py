"""Test script for Conversation Detection system."""

import logging
import sys
import time
import threading
from queue import Queue
import collections
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sounddevice as sd

# Add src directory to Python path
src_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(src_dir))

from src.audio.capture import AudioCapture
from src.vad.processor import VADProcessor, VADConfig
from src.conversation.detector import ConversationDetector
from src.messages import ConversationMessage

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AudioPlayer:
    """Plays audio chunks in real-time using sounddevice."""
    
    def __init__(self, sample_rate=44000, channels=1, chunk_size=512):
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.is_muted = False
        self.audio_buffer = collections.deque(maxlen=100)
        self.playback_queue = Queue()  # Queue for conversation playback
        self.is_playing_conversation = False
        
        def audio_callback(outdata: np.ndarray, frames: int, time_info: dict, status: sd.CallbackFlags) -> None:
            if status:
                logger.warning(f"Audio output status: {status}")
            
            if not self.is_muted and not self.is_playing_conversation and self.audio_buffer:
                # Normal live playback
                try:
                    data = self.audio_buffer.popleft()
                    if len(data) < len(outdata):
                        data = np.pad(data, (0, len(outdata) - len(data)))
                    outdata[:] = data.reshape(-1, 1)
                except Exception as e:
                    logger.error(f"Error in audio callback: {e}")
                    outdata.fill(0)
            elif self.is_playing_conversation:
                # Playback recorded conversation
                try:
                    if not self.playback_queue.empty():
                        data = self.playback_queue.get_nowait()
                        if len(data) < len(outdata):
                            data = np.pad(data, (0, len(outdata) - len(data)))
                        outdata[:] = data.reshape(-1, 1)
                    else:
                        self.is_playing_conversation = False
                        outdata.fill(0)
                except Exception as e:
                    logger.error(f"Error in conversation playback: {e}")
                    outdata.fill(0)
            else:
                outdata.fill(0)
        
        self.stream = sd.OutputStream(
            channels=channels,
            samplerate=sample_rate,
            callback=audio_callback,
            blocksize=chunk_size
        )
        self.stream.start()
    
    def play(self, chunk: np.ndarray) -> None:
        try:
            self.audio_buffer.append(chunk)
        except Exception as e:
            logger.error(f"Error adding audio to playback buffer: {e}")
    
    def play_conversation(self, audio_data: np.ndarray) -> None:
        """Play back a recorded conversation."""
        if not self.is_muted:
            return
        
        try:
            logger.info("Playing back conversation...")
            self.is_playing_conversation = True
            
            # Split audio into chunks
            chunk_samples = self.chunk_size
            for i in range(0, len(audio_data), chunk_samples):
                chunk = audio_data[i:i + chunk_samples]
                self.playback_queue.put(chunk)
        except Exception as e:
            logger.error(f"Error queueing conversation playback: {e}")
            self.is_playing_conversation = False
    
    def toggle_mute(self) -> None:
        self.is_muted = not self.is_muted
        logger.info(f"Audio {'muted' if self.is_muted else 'unmuted'}")
    
    def stop(self) -> None:
        if hasattr(self, 'stream') and self.stream:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception as e:
                logger.error(f"Error stopping audio stream: {e}")

class ConversationVisualizer:
    """Real-time visualization of Conversation Detection."""
    
    def __init__(self, window_size: int = 2000):
        self.window_size = window_size
        self.running = True
        
        # Data storage with timestamps
        self.timestamps = collections.deque(maxlen=window_size)
        self.audio_data = collections.deque(maxlen=window_size)
        self.is_speech = collections.deque(maxlen=window_size)
        self.is_conversation = collections.deque(maxlen=window_size)
        
        # Initialize with zeros
        current_time = time.time()
        for i in range(window_size):
            self.timestamps.append(current_time - (window_size - i) * 0.02)  # Assuming 20ms chunks
            self.audio_data.append(0)
            self.is_speech.append(0)
            self.is_conversation.append(0)
        
        # Set up plot
        plt.style.use('dark_background')
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(12, 8))
        self.fig.suptitle('Conversation Detection', fontsize=14)
        
        # Audio waveform
        self.line_audio, = self.ax1.plot([], [], 'cyan', label='Audio', linewidth=1)
        self.ax1.set_ylim(-1, 1)
        self.ax1.set_xlabel('Time (s)')
        self.ax1.set_ylabel('Amplitude')
        self.ax1.grid(True, alpha=0.3)
        self.ax1.legend(loc='upper right')
        
        # Speech and conversation state
        self.line_speech, = self.ax2.plot([], [], 'yellow', label='Speech', linewidth=1, alpha=0.7)
        self.line_conv, = self.ax2.plot([], [], 'red', label='Conversation', linewidth=2)
        self.ax2.set_ylim(-0.1, 1.1)
        self.ax2.set_xlabel('Time (s)')
        self.ax2.set_ylabel('State')
        self.ax2.grid(True, alpha=0.3)
        self.ax2.legend(loc='upper right')
        
        # Status text
        self.status_text = self.fig.text(
            0.02, 0.02,
            'Press Q to quit\nPress M to toggle audio',
            color='white',
            alpha=0.7
        )
        
        # Audio player reference
        self.audio_player = None
        
        # Lock for thread safety
        self.data_lock = threading.Lock()
        
        # Animation
        self.anim = None
        
        # Window events
        self._close_cid = self.fig.canvas.mpl_connect('close_event', self._on_close)
        self._key_cid = self.fig.canvas.mpl_connect('key_press_event', self._on_key)
    
    def _update(self, frame):
        with self.data_lock:
            # Update time window
            now = time.time()
            window_start = now - 10  # Show last 10 seconds
            
            # Filter data within time window
            times = np.array(self.timestamps)
            mask = times >= window_start
            plot_times = times[mask] - now  # Relative to current time
            
            # Update audio plot
            audio = np.array(self.audio_data)[mask]
            self.line_audio.set_data(plot_times, audio)
            self.ax1.set_xlim(min(plot_times), max(plot_times))
            
            # Update speech/conversation plot
            speech = np.array(self.is_speech)[mask]
            conv = np.array(self.is_conversation)[mask]
            self.line_speech.set_data(plot_times, speech)
            self.line_conv.set_data(plot_times, conv)
            self.ax2.set_xlim(min(plot_times), max(plot_times))
        
        return self.line_audio, self.line_speech, self.line_conv
    
    def add_data(self, timestamp: float, audio_chunk: np.ndarray, vad_state, is_in_conversation: bool):
        with self.data_lock:
            # Add new data points
            for sample in audio_chunk:
                self.timestamps.append(timestamp)
                self.audio_data.append(sample)
                self.is_speech.append(1 if vad_state.is_speech else 0)
                self.is_conversation.append(1 if is_in_conversation else 0)
    
    def _on_key(self, event):
        if event.key == 'q':
            self.running = False
            plt.close(self.fig)
        elif event.key == 'm' and self.audio_player:
            self.audio_player.toggle_mute()
    
    def _on_close(self, event):
        self.running = False
    
    def cleanup(self):
        if self.anim:
            self.anim.event_source.stop()
        plt.close(self.fig)
    
    def start_animation(self):
        self.anim = FuncAnimation(
            self.fig,
            self._update,
            interval=50,  # 20 FPS
            blit=True
        )
    
    def set_audio_player(self, player):
        self.audio_player = player
    
    def show(self):
        plt.show()

def main():
    """Main test function."""
    # Initialize components to None for cleanup
    audio_capture = None
    conversation_detector = None
    audio_player = None
    visualizer = None
    
    try:
        # Initialize components
        audio_capture = AudioCapture()
        vad_processor = VADProcessor(VADConfig())
        audio_player = AudioPlayer(sample_rate=audio_capture.sample_rate)
        visualizer = ConversationVisualizer()
        
        # Set up conversation detector
        def on_conversation(message: ConversationMessage):
            logger.info(
                f"Conversation detected: {message.start_time:.2f}s - {message.end_time:.2f}s "
                f"({len(message.speech_segments)} speech segments)"
            )
            # Play back conversation if muted
            audio_player.play_conversation(message.audio_data)
        
        conversation_detector = ConversationDetector(
            audio_capture,
            vad_processor,
            on_conversation=on_conversation
        )
        
        # Set up audio callback
        def handle_audio(timestamp: float, original: np.ndarray, downsampled: np.ndarray):
            # Only play live audio if not playing back a conversation
            if not audio_player.is_playing_conversation:
                audio_player.play(original)
            
            # Update VAD
            vad_processor.add_audio(downsampled)
            vad_state = vad_processor.get_state()
            
            # Update visualization
            is_in_conversation = conversation_detector._conversation_start is not None
            visualizer.add_data(timestamp, original, vad_state, is_in_conversation)
        
        # Connect components
        audio_capture.add_callback(handle_audio)
        visualizer.set_audio_player(audio_player)
        
        # Start components
        audio_capture.start()
        conversation_detector.start()
        
        # Start visualization
        visualizer.start_animation()
        visualizer.show()
        
    except Exception as e:
        logger.error(f"Error in main: {e}", exc_info=True)
    
    finally:
        # Clean up components if they were initialized
        if conversation_detector:
            try:
                conversation_detector.stop()
            except Exception as e:
                logger.error(f"Error stopping conversation detector: {e}")
        
        if audio_capture:
            try:
                audio_capture.stop()
            except Exception as e:
                logger.error(f"Error stopping audio capture: {e}")
        
        if audio_player:
            try:
                audio_player.stop()
            except Exception as e:
                logger.error(f"Error stopping audio player: {e}")
        
        if visualizer:
            try:
                visualizer.cleanup()
            except Exception as e:
                logger.error(f"Error cleaning up visualizer: {e}")

if __name__ == "__main__":
    main()

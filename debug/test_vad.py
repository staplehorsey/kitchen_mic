"""Test script for Voice Activity Detection."""

import logging
import sys
import time
import threading
from queue import Queue, Full, Empty
from typing import Optional
import collections
from threading import Lock, Thread
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sounddevice as sd

# Add src directory to Python path
src_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(src_dir))

from src.vad.processor import VADProcessor, VADConfig
from src.audio.capture import AudioCapture

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AudioPlayer:
    """Plays audio chunks in real-time using sounddevice."""
    
    def __init__(self, sample_rate=44000, channels=1, chunk_size=512):
        """Initialize audio player."""
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.is_muted = False
        self.audio_buffer = collections.deque(maxlen=100)  # Buffer ~3 seconds
        
        def audio_callback(outdata: np.ndarray, frames: int, time_info: dict, status: sd.CallbackFlags) -> None:
            """Process audio data from stream."""
            if status:
                logger.warning(f"Audio output status: {status}")
            
            if self.is_muted or not self.audio_buffer:
                outdata.fill(0)
                return
            
            try:
                # Get audio data from buffer
                data = self.audio_buffer.popleft()
                if len(data) < len(outdata):
                    # Pad with zeros if not enough data
                    data = np.pad(data, (0, len(outdata) - len(data)))
                outdata[:] = data.reshape(-1, 1)
            except Exception as e:
                logger.error(f"Error in audio callback: {e}")
                outdata.fill(0)
        
        # Open stream with callback
        self.stream = sd.OutputStream(
            channels=channels,
            samplerate=sample_rate,
            callback=audio_callback,
            blocksize=chunk_size
        )
        self.stream.start()
    
    def play(self, chunk: np.ndarray) -> None:
        """Add audio chunk to playback buffer."""
        try:
            self.audio_buffer.append(chunk)
        except Exception as e:
            logger.error(f"Error adding audio to playback buffer: {e}")
    
    def toggle_mute(self) -> None:
        """Toggle audio mute state."""
        self.is_muted = not self.is_muted
        logger.info(f"Audio {'muted' if self.is_muted else 'unmuted'}")
    
    def stop(self) -> None:
        """Stop audio playback."""
        if hasattr(self, 'stream') and self.stream:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception as e:
                logger.error(f"Error stopping audio stream: {e}")

class VADVisualizer:
    """Real-time visualization of Voice Activity Detection."""
    
    def __init__(self, window_size: int = 2000):
        """Initialize visualizer.
        
        Args:
            window_size: Number of samples to show in visualization
        """
        self.window_size = window_size
        self.running = True
        
        # Data storage
        self.audio_data = collections.deque(maxlen=window_size)
        self.speech_probs = collections.deque(maxlen=window_size)
        self.is_speech = collections.deque(maxlen=window_size)
        self.conversation_timeline = []
        
        # Initialize with zeros
        for _ in range(window_size):
            self.audio_data.append(0)
            self.speech_probs.append(0)
            self.is_speech.append(0)
        
        # Set up plot
        plt.style.use('dark_background')
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(3, 1, figsize=(12, 10))
        self.fig.suptitle('Voice Activity Detection', fontsize=14)
        
        # Audio waveform
        self.line_audio, = self.ax1.plot([], [], 'cyan', label='Audio', linewidth=1)
        self.ax1.set_ylim(-1, 1)
        self.ax1.set_xlabel('Time (s)')
        self.ax1.set_ylabel('Amplitude')
        self.ax1.grid(True, alpha=0.3)
        self.ax1.legend(loc='upper right')
        
        # VAD probability
        self.line_prob, = self.ax2.plot([], [], 'lime', label='Speech Prob', linewidth=2)
        self.line_speech, = self.ax2.plot([], [], 'red', label='Speech', linewidth=1, alpha=0.7)
        self.ax2.set_ylim(-0.1, 1.1)
        self.ax2.set_xlabel('Time (s)')
        self.ax2.set_ylabel('Probability')
        self.ax2.grid(True, alpha=0.3)
        self.ax2.legend(loc='upper right')
        
        # Conversation timeline
        self.ax3.set_ylim(-0.1, 1.1)
        self.ax3.set_xlabel('Time (s)')
        self.ax3.set_ylabel('Conversation')
        self.ax3.grid(True, alpha=0.3)
        
        # Status text
        self.status_text = self.fig.text(
            0.02, 0.02,
            'Press Q to quit\nPress M to toggle audio',
            color='white',
            alpha=0.7
        )
        
        # Audio player reference for mute toggle
        self.audio_player = None
        
        # Lock for thread safety
        self.data_lock = threading.Lock()
        
        # Animation
        self.anim = None
        
        # Window close event
        self.fig.canvas.mpl_connect('close_event', self._on_close)
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)
    
    def start_animation(self):
        """Start the animation."""
        self.anim = FuncAnimation(
            self.fig,
            self._update,
            interval=50,
            blit=True
        )
    
    def set_audio_player(self, player):
        """Set audio player reference for mute toggle."""
        self.audio_player = player
    
    def add_data(self, audio_chunk: np.ndarray, vad_state):
        """Add new data to visualization."""
        with self.data_lock:
            # Update audio data (keep last N samples)
            self.audio_data.extend(audio_chunk)
            
            # Update speech probability and detection
            self.speech_probs.append(float(vad_state.speech_probability))
            self.is_speech.append(1.0 if vad_state.is_speech else 0.0)
            
            # Update conversation timeline
            if vad_state.conversation_start is not None:
                # Only add new conversation if it doesn't overlap with the last one
                if not self.conversation_timeline or self.conversation_timeline[-1][0] != vad_state.conversation_start:
                    self.conversation_timeline.append((vad_state.conversation_start, vad_state.conversation_end))
            
            # Remove old conversations
            current_time = time.time()
            window_start = current_time - (self.window_size / 20)  # Keep last N seconds
            self.conversation_timeline = [
                (start, end) for start, end in self.conversation_timeline 
                if end is None or end > window_start
            ]
    
    def _update(self, frame):
        """Update animation frame."""
        with self.data_lock:
            current_time = time.time()
            
            # Create time arrays
            times = np.linspace(
                current_time - self.window_size / 20,  # Start time
                current_time,  # End time
                len(self.audio_data)  # Number of points
            )
            
            # Update audio waveform
            audio_data = np.array(list(self.audio_data))
            self.line_audio.set_data(times, audio_data)
            self.ax1.set_xlim(times[0], times[-1])
            
            # Update speech probability
            prob_times = np.linspace(
                current_time - self.window_size / 20,
                current_time,
                len(self.speech_probs)
            )
            probs = np.array(list(self.speech_probs))
            speech = np.array(list(self.is_speech))
            
            self.line_prob.set_data(prob_times, probs)
            self.line_speech.set_data(prob_times, speech)
            self.ax2.set_xlim(times[0], times[-1])
            
            # Update conversation timeline
            # Remove old spans
            for collection in self.ax3.collections[:]:
                collection.remove()
            
            # Add new spans
            for start, end in self.conversation_timeline:
                if start > 0:  # Only show actual conversations
                    end_time = end if end is not None else current_time
                    self.ax3.axvspan(start, end_time, color='blue', alpha=0.2)
            self.ax3.set_xlim(times[0], times[-1])
        
        return self.line_audio, self.line_prob, self.line_speech
    
    def _on_close(self, event):
        """Handle window close event."""
        self.running = False
    
    def _on_key(self, event):
        """Handle key press events."""
        if event.key == 'q':
            plt.close('all')
            self.running = False
        elif event.key == 'm' and self.audio_player:
            self.audio_player.toggle_mute()
    
    def show(self):
        """Show visualization window."""
        plt.show()

def main():
    """Main test function."""
    vad = None
    capture = None
    player = None
    viz = None
    
    try:
        # Initialize components
        vad = VADProcessor()
        viz = VADVisualizer()
        player = AudioPlayer()
        viz.set_audio_player(player)
        
        # Start audio capture
        def on_data(original_chunk: np.ndarray, downsampled_chunk: np.ndarray):
            """Process captured audio data."""
            try:
                if vad and viz and viz.running:
                    # Add audio for VAD processing
                    vad.add_audio(downsampled_chunk)
                    
                    # Get current VAD state
                    state = vad.get_state()
                    
                    # Update visualization
                    viz.add_data(original_chunk, state)
                    
                    # Play audio
                    if player:
                        player.play(original_chunk)
                
            except Exception as e:
                logger.error(f"Error in audio callback: {e}")
        
        capture = AudioCapture(on_data=on_data)
        capture.start()
        
        # Start visualization
        viz.start_animation()
        viz.show()
        
    except Exception as e:
        logger.error(f"Error in main: {e}")
    finally:
        # Cleanup
        logger.info("Cleaning up...")
        if vad:
            try:
                vad.stop()
            except Exception as e:
                logger.error(f"Error stopping VAD: {e}")
        
        if capture:
            try:
                capture.stop()
            except Exception as e:
                logger.error(f"Error stopping capture: {e}")
        
        if player:
            try:
                player.stop()
            except Exception as e:
                logger.error(f"Error stopping audio player: {e}")
        
        if viz:
            try:
                viz.running = False
                plt.close('all')
            except Exception as e:
                logger.error(f"Error cleaning up visualization: {e}")

if __name__ == "__main__":
    main()

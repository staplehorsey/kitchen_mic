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
import datetime
from matplotlib.lines import Line2D

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Patch
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

from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)
class TimeRange:
    start: float
    end: float
    
    def contains(self, time: float) -> bool:
        return self.start <= time <= self.end

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
        """Initialize visualizer."""
        self.window_size = window_size
        self.running = True
        
        # Data storage
        self.audio_data = collections.deque(maxlen=window_size)
        self.speech_probs = collections.deque(maxlen=window_size)
        self.is_speech = collections.deque(maxlen=window_size)
        self.start_time = time.time()
        
        # Initialize with zeros
        for _ in range(window_size):
            self.audio_data.append(0)
            self.speech_probs.append(0)
            self.is_speech.append(0)
        
        # Set up plot
        plt.style.use('dark_background')
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(12, 8))
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
        
        # Queue for visualization updates
        self.update_queue = Queue()
        
        # Animation
        self.anim = None
        
        # Window close event
        self._close_cid = self.fig.canvas.mpl_connect('close_event', self._on_close)
        self._key_cid = self.fig.canvas.mpl_connect('key_press_event', self._on_key)
    
    def _update(self, frame):
        """Update animation frame."""
        try:
            with self.data_lock:
                # Calculate relative time and time windows
                relative_time = time.time() - self.start_time
                times = np.linspace(
                    relative_time - self.window_size / 20,
                    relative_time,
                    len(self.audio_data)
                )
                
                # Update audio waveform
                self.line_audio.set_data(times, list(self.audio_data))
                self.ax1.set_xlim(times[0], times[-1])
                
                # Update speech probability
                prob_times = np.linspace(
                    relative_time - self.window_size / 20,
                    relative_time,
                    len(self.speech_probs)
                )
                probs = np.array(list(self.speech_probs))
                speech = np.array(list(self.is_speech))
                
                self.line_prob.set_data(prob_times, probs)
                self.line_speech.set_data(prob_times, speech)
                self.ax2.set_xlim(times[0], times[-1])
                
                if len(probs) > 0:
                    logger.debug(f"Latest speech prob: {probs[-1]:.2f}, is_speech: {speech[-1]}")
                
                self.fig.canvas.draw_idle()
                self.fig.canvas.flush_events()
                
        except Exception as e:
            logger.error(f"Error in update: {e}")
    
    def add_data(self, audio_chunk: np.ndarray, vad_state):
        """Add new data to visualization."""
        try:
            with self.data_lock:
                # Update audio data
                self.audio_data.extend(audio_chunk)
                self.speech_probs.append(float(vad_state.speech_probability))
                self.is_speech.append(1.0 if vad_state.is_speech else 0.0)
                
        except Exception as e:
            logger.error(f"Error adding data: {e}")
    
    def _on_key(self, event):
        """Handle key press events."""
        if event.key == 'q':
            plt.close(self.fig)
        elif event.key == 'm' and self.audio_player:
            self.audio_player.toggle_mute()
    
    def _on_close(self, event):
        """Handle window close event."""
        self.running = False
        self.cleanup()
    
    def cleanup(self):
        """Clean up resources."""
        if self.anim:
            self.anim.event_source.stop()
        if hasattr(self, '_close_cid') and self._close_cid:
            self.fig.canvas.mpl_disconnect(self._close_cid)
        if hasattr(self, '_key_cid') and self._key_cid:
            self.fig.canvas.mpl_disconnect(self._key_cid)
    
    def start_animation(self):
        """Start the animation."""
        self.anim = FuncAnimation(
            self.fig,
            self._update,
            interval=50,  # Update every 50ms
            blit=False,
            cache_frame_data=False
        )
    
    def set_audio_player(self, player):
        """Set audio player reference for mute toggle."""
        self.audio_player = player
    
    def show(self):
        """Show visualization window."""
        plt.show()

def main():
    """Main test function."""
    # Set up logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    vad = None
    capture = None
    player = None
    viz = None
    
    try:
        # Initialize components
        vad = VADProcessor(VADConfig())
        viz = VADVisualizer()
        
        # Create audio capture first to get sample rate
        capture = AudioCapture(host="staple.local")
        player = AudioPlayer(sample_rate=capture.sample_rate)
        viz.set_audio_player(player)
        
        # Set up audio callback
        def handle_audio(timestamp: float, original: np.ndarray, downsampled: np.ndarray):
            """Process captured audio data."""
            try:
                # Add audio to VAD
                vad.add_audio(downsampled)
                
                # Get VAD state
                state = vad.get_state()
                
                # Update visualization
                viz.add_data(original, state)
                
                # Play audio
                player.play(original)
                
            except Exception as e:
                logger.error(f"Error processing audio: {e}")
        
        # Register callback and start capture
        capture.add_callback(handle_audio)
        capture.start()
        
        # Start visualization
        viz.start_animation()
        viz.show()
        
    except Exception as e:
        logger.error(f"Error in main: {e}", exc_info=True)
    finally:
        logger.info("Cleaning up...")
        # Clean up components
        if capture:
            try:
                capture.stop()
            except Exception as e:
                logger.error(f"Error stopping capture: {e}")
        
        if vad:
            try:
                vad.stop()
            except Exception as e:
                logger.error(f"Error stopping VAD: {e}")
        
        if player:
            try:
                player.stop()
            except Exception as e:
                logger.error(f"Error stopping player: {e}")
        
        if viz:
            try:
                viz.cleanup()
            except Exception as e:
                logger.error(f"Error cleaning up visualizer: {e}")

if __name__ == "__main__":
    main()

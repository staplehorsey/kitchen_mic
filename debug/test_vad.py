"""Test script for Voice Activity Detection."""

import logging
import sys
import time
from pathlib import Path
from queue import Queue, Empty, Full
from threading import Event, Thread, Lock
import collections
from typing import Tuple
import threading
import sounddevice as sd

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Add src to Python path
src_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(src_dir))

from src.audio.capture import AudioCapture
from src.vad.processor import VADProcessor, VADConfig

# Configure logging - only show INFO and above for most modules
logging.getLogger().setLevel(logging.INFO)
for handler in logging.getLogger().handlers:
    handler.setLevel(logging.INFO)

# Module logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Only show debug logs for specific modules
logging.getLogger('src.vad.processor').setLevel(logging.DEBUG)

class AudioBuffer:
    """Thread-safe circular audio buffer."""
    
    def __init__(self, max_size: int = 44100):
        """Initialize buffer.
        
        Args:
            max_size: Maximum number of samples to store
        """
        self.buffer = collections.deque(maxlen=max_size)
        self.timestamps = collections.deque(maxlen=max_size)
        self.lock = Lock()
        
    def add(self, data: np.ndarray, timestamp: float):
        """Add samples to buffer with timestamp."""
        with self.lock:
            for sample in data:
                self.buffer.append(sample)
                self.timestamps.append(timestamp)
    
    def get(self, n: int = None) -> Tuple[np.ndarray, float]:
        """Get samples from buffer with timestamp.
        
        Args:
            n: Number of samples to get. If None, get all.
            
        Returns:
            Tuple of (samples, timestamp) where samples is a numpy array
            and timestamp is the time of the first sample
        """
        with self.lock:
            if n is None:
                n = len(self.buffer)
            if n > len(self.buffer):
                n = len(self.buffer)
            
            # Get samples and timestamps
            samples = list(self.buffer)[-n:]
            timestamps = list(self.timestamps)[-n:]
            
            return np.array(samples, dtype=np.float32), timestamps[0] if timestamps else 0.0

class AudioPlayer:
    """Plays audio chunks in real-time using sounddevice."""
    
    def __init__(self, sample_rate=16000, channels=1, chunk_size=512):
        """Initialize audio player."""
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.is_muted = False
        self.audio_buffer = collections.deque(maxlen=100)  # Buffer ~3 seconds
        
        def audio_callback(outdata, frames, time, status):
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
    
    def play(self, chunk: np.ndarray, timestamp: float):
        """Add audio chunk to playback buffer."""
        if not self.is_muted:
            try:
                self.audio_buffer.append(chunk.astype(np.float32))
            except Exception as e:
                logger.error(f"Error queueing audio: {e}")
    
    def toggle_mute(self):
        """Toggle audio mute state."""
        self.is_muted = not self.is_muted
        if self.is_muted:
            # Clear buffer when muting
            self.audio_buffer.clear()
        logger.debug(f"Audio {'muted' if self.is_muted else 'unmuted'}")
    
    def close(self):
        """Close audio player."""
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None

class AudioProcessor(Thread):
    """Background thread for audio processing."""
    
    def __init__(self, capture, player, vad, viz):
        super().__init__()
        self.capture = capture
        self.player = player
        self.vad = vad
        self.viz = viz
        self.running = True
        self.daemon = True
        
        # Processing state
        self.last_prob = 0.0
        self.last_speech = False
        self.decay_rate = 0.05
        
        # Create separate thread for VAD processing
        self.vad_queue = Queue(maxsize=100)
        self.vad_thread = Thread(target=self._vad_loop, daemon=True)
        self.vad_thread.start()
    
    def _vad_loop(self):
        """Background thread for VAD processing."""
        while self.running:
            try:
                # Get chunk from queue
                chunk = self.vad_queue.get(timeout=0.1)
                
                # Process VAD
                segments = self.vad.process_audio(chunk)
                
                # Update state
                if segments:
                    self.last_speech = True
                    self.last_prob = 1.0
                else:
                    self.last_speech = False
                    self.last_prob = max(0.0, self.last_prob - self.decay_rate)
                
                # Update visualization
                self.viz.add_data(chunk, self.last_prob, self.last_speech, 0)
                
            except Empty:
                continue
            except Exception as e:
                logger.error(f"Error in VAD processing: {e}")
                time.sleep(0.1)
    
    def run(self):
        """Main processing loop."""
        logger.debug("Audio processor thread started")
        
        # Audio chunk size must match VAD window size
        chunk_size = self.vad.config.window_size_samples
        last_process_time = time.time()
        
        while self.running and self.viz.running:
            try:
                # Get audio data
                original, downsampled = self.capture.get_audio_data()
                if len(downsampled) > 0:
                    # Process in VAD window-sized chunks
                    for i in range(0, len(downsampled), chunk_size):
                        chunk = downsampled[i:i + chunk_size]
                        if len(chunk) == chunk_size:
                            # Queue for VAD processing with backpressure
                            try:
                                if not self.vad_queue.full():
                                    self.vad_queue.put_nowait(chunk)
                            except Full:
                                # Skip VAD if queue is full but continue playback
                                pass
                            
                            # Send to audio player
                            self.player.play(chunk, 0)
                
                # Adaptive sleep based on processing time
                process_time = time.time() - last_process_time
                sleep_time = max(0.01, 0.05 - process_time)  # Target 50ms cycle
                time.sleep(sleep_time)
                last_process_time = time.time()
                
            except Exception as e:
                logger.error(f"Error in audio processing loop: {e}")
                time.sleep(0.1)
        
        logger.debug("Audio processor thread stopped")
    
    def stop(self):
        """Stop the processing thread."""
        self.running = False
        if self.vad_thread.is_alive():
            self.vad_thread.join(timeout=1.0)
        self.join(timeout=1.0)

class VADVisualizer:
    """Real-time visualization of Voice Activity Detection."""
    
    def __init__(self, window_size: int = 2000, prob_smoothing: int = 20):
        """Initialize visualizer.
        
        Args:
            window_size: Number of samples to show in visualization
            prob_smoothing: Number of frames to average for probability smoothing
        """
        self.window_size = window_size
        self.prob_smoothing = prob_smoothing
        self.running = True
        
        # Data storage with timestamps
        self.audio_data = np.zeros(window_size)
        self.speech_probs = np.zeros(window_size)
        self.is_speech = np.zeros(window_size, dtype=bool)
        
        # Probability smoothing
        self.prob_buffer = collections.deque(maxlen=prob_smoothing)
        
        # Set up plot
        plt.style.use('dark_background')
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(12, 8))
        self.fig.suptitle('Voice Activity Detection', fontsize=14)
        
        # Audio waveform
        self.line_audio, = self.ax1.plot(self.audio_data, 'cyan', label='Audio', linewidth=1)
        self.ax1.set_ylim(-1, 1)
        self.ax1.set_xlabel('Time (s)')
        self.ax1.set_ylabel('Amplitude')
        self.ax1.grid(True, alpha=0.3)
        self.ax1.legend(loc='upper right')
        
        # VAD probability
        self.line_prob, = self.ax2.plot(self.speech_probs, 'lime', label='Speech Prob', linewidth=2)
        self.line_speech, = self.ax2.plot(self.is_speech, 'red', label='Speech', linewidth=1, alpha=0.7)
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
        
        # Timing
        self.start_time = time.time()
        
        # Window close event
        self.fig.canvas.mpl_connect('close_event', self._on_close)
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)
        
        # Animation - create after all attributes are set
        self.anim = None
        
    def start_animation(self):
        """Start the animation."""
        # Update every 50ms (20 FPS)
        self.anim = FuncAnimation(
            self.fig, self._update, interval=50,
            blit=True, cache_frame_data=False
        )
        
    def set_audio_player(self, player):
        """Set audio player reference for mute toggle."""
        self.audio_player = player
        self.start_animation()  # Start animation after player is set
        
    def add_data(self, audio_chunk: np.ndarray, prob: float, is_speech: bool, timestamp: float):
        """Add new data to visualization."""
        # Shift existing data left
        self.audio_data = np.roll(self.audio_data, -len(audio_chunk))
        self.speech_probs = np.roll(self.speech_probs, -len(audio_chunk))
        self.is_speech = np.roll(self.is_speech, -len(audio_chunk))
        
        # Add new data at the end
        self.audio_data[-len(audio_chunk):] = audio_chunk
        self.speech_probs[-len(audio_chunk):] = prob
        self.is_speech[-len(audio_chunk):] = is_speech
        
    def _update(self, frame):
        """Update animation frame."""
        self.line_audio.set_ydata(self.audio_data)
        self.line_prob.set_ydata(self.speech_probs)
        self.line_speech.set_ydata(self.is_speech)
        return self.line_audio, self.line_prob, self.line_speech
    
    def _on_close(self, event):
        """Handle window close event."""
        logger.debug("Window close event received")
        self.running = False
        plt.close('all')
    
    def _on_key(self, event):
        """Handle key press events."""
        if event.key == 'q':
            logger.debug("Quit key pressed")
            self.running = False
            plt.close('all')
        elif event.key == 'm':
            if self.audio_player:
                self.audio_player.toggle_mute()
    
    def show(self):
        """Show visualization window."""
        plt.show()

def main():
    """Main test function."""
    viz = None
    try:
        # Initialize components
        logger.info("Initializing components...")
        
        # Audio capture
        capture = AudioCapture(
            channels=1,
            chunk_size=512,
            original_rate=44000,
            target_rate=16000
        )
        
        # Audio playback
        player = AudioPlayer()
        
        # Voice activity detection
        vad_config = VADConfig(
            threshold=0.5,
            min_speech_duration_ms=250,
            min_silence_duration_ms=100,
            speech_pad_ms=30
        )
        
        vad = VADProcessor(config=vad_config)
        
        # Visualization
        viz = VADVisualizer(prob_smoothing=5)
        viz.set_audio_player(player)  # Set player reference for mute toggle
        
        # Audio processor thread
        processor = AudioProcessor(capture, player, vad, viz)
        processor.start()
        
        # Start audio capture
        capture.start()
        
        # Show visualization (blocks until window closed)
        try:
            viz.show()
        except KeyboardInterrupt:
            logger.info("Test interrupted by user")
        finally:
            logger.info("Test complete")
            logger.info("Cleaning up...")
            
            # Stop components
            processor.stop()
            capture.stop()
            player.close()
            
            logger.info("Test complete")
            
    except Exception as e:
        logger.error(f"Test failed: {e}")
        if viz:
            viz.running = False
        raise

if __name__ == "__main__":
    main()

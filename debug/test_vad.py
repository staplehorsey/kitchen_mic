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

@dataclass(frozen=True)
class ConversationState:
    speech_start: float
    conversation_start: Optional[float] = None
    speech_end: Optional[float] = None
    conversation_end: Optional[float] = None
    
    def get_ranges(self, current_time: float):
        """Get all the ranges and markers for this conversation state."""
        ranges = []
        markers = []
        
        # Speech start marker
        markers.append(('blue', self.speech_start, 'Speech Start'))
        
        # Initial speech range (yellow)
        if self.conversation_start:
            ranges.append(('yellow', TimeRange(self.speech_start, self.conversation_start)))
            # Conversation start marker
            markers.append(('green', self.conversation_start, 'Conversation Start'))
            
            # Active conversation range (green)
            if self.speech_end:
                ranges.append(('green', TimeRange(self.conversation_start, self.speech_end)))
        else:
            # If no conversation started, yellow continues until speech end or current time
            end_time = self.speech_end if self.speech_end else current_time
            ranges.append(('yellow', TimeRange(self.speech_start, end_time)))
        
        # Speech end marker and post-speech range
        if self.speech_end:
            markers.append(('red', self.speech_end, 'Speech End'))
            
            # Post speech range (orange)
            conv_end_time = self.speech_end + 15
            if current_time < conv_end_time:
                ranges.append(('orange', TimeRange(self.speech_end, min(current_time, conv_end_time))))
            else:
                # Conversation end marker
                markers.append(('purple', conv_end_time, 'Conversation End'))
        
        return ranges, markers

    @staticmethod
    def from_dict(conv_dict: dict, current_time: float) -> 'ConversationState':
        """Create a ConversationState from a conversation dictionary."""
        state = ConversationState(
            speech_start=conv_dict['speech_start'],
            conversation_start=conv_dict.get('conversation_start'),
            speech_end=conv_dict.get('last_speech'),
            conversation_end=None  # We'll calculate this based on timing
        )
        
        # If speech has ended and enough time has passed, set conversation_end
        if state.speech_end and (current_time - state.speech_end) >= 15:
            # Create new state with conversation_end set
            return ConversationState(
                speech_start=state.speech_start,
                conversation_start=state.conversation_start,
                speech_end=state.speech_end,
                conversation_end=state.speech_end + 15
            )
        
        return state

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
        self.conversations = []  # List of conversation events
        self.current_conversation = None
        self.timeline_start = None
        self.start_time = time.time()
        
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
        self.ax3.set_xlabel('Time')
        self.ax3.set_ylabel('Conversation State')
        self.ax3.grid(True, alpha=0.3)
        
        # Add legend for conversation states
        self.legend_elements = [
            Line2D([0], [0], color='blue', linestyle='-', linewidth=2, label='Speech Start'),
            Line2D([0], [0], color='green', linestyle='-', linewidth=2, label='Conversation Start'),
            Line2D([0], [0], color='red', linestyle='-', linewidth=2, label='Speech End'),
            Line2D([0], [0], color='purple', linestyle='-', linewidth=3, label='Conversation End'),
            Patch(facecolor='yellow', alpha=0.5, label='Initial Speech'),
            Patch(facecolor='green', alpha=0.3, label='Conversation Active'),
            Patch(facecolor='orange', alpha=0.4, label='Post-Speech')
        ]
        self.ax3.legend(handles=self.legend_elements, loc='upper right')
        
        # Status text
        self.status_text = self.fig.text(
            0.02, 0.02,
            'Press Q to quit\nPress M to toggle audio\nPress R to reset timeline',
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
    
    def reset_timeline(self):
        """Reset the conversation timeline."""
        with self.data_lock:
            self.conversations = []
            self.current_conversation = None
            self.timeline_start = None
            self.start_time = time.time()
            self._update_conversation_timeline()
    
    def _format_timestamp(self, timestamp: float) -> str:
        """Format timestamp as HH:MM:SS."""
        dt = datetime.datetime.fromtimestamp(timestamp)
        return dt.strftime('%H:%M:%S')
        
    def _update(self, frame):
        """Update animation frame."""
        logger.debug("\n=== Animation Update Frame ===")
        start_time = time.time()
        
        # Process any pending visualization updates
        updates_processed = 0
        while True:
            try:
                update_func = self.update_queue.get_nowait()
                update_func()
                updates_processed += 1
            except Empty:
                break
        
        if updates_processed > 0:
            logger.debug(f"Processed {updates_processed} visualization updates")
            
        with self.data_lock:
            # Update audio waveform
            relative_time = time.time() - self.start_time
            times = np.linspace(
                relative_time - self.window_size / 20,
                relative_time,
                len(self.audio_data)
            )
            
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
        
        end_time = time.time()
        logger.debug(f"Frame update took {(end_time - start_time)*1000:.1f}ms")
        
        # Don't return artists when not using blit
        return []
    
    def add_data(self, audio_chunk: np.ndarray, vad_state):
        """Add new data to visualization."""
        with self.data_lock:
            # Update audio data
            self.audio_data.extend(audio_chunk)
            self.speech_probs.append(float(vad_state.speech_probability))
            self.is_speech.append(1.0 if vad_state.is_speech else 0.0)
            
            # Check for conversation state changes
            current_time = time.time()
            timeline_updated = False
            
            logger.debug("\n=== VAD State Update ===")
            logger.debug(f"Current time: {self._format_timestamp(current_time)} ({current_time})")
            logger.debug(f"VAD State:")
            logger.debug(f"  speech_prob: {vad_state.speech_probability:.2f}")
            logger.debug(f"  is_speech: {vad_state.is_speech}")
            logger.debug(f"  first_speech: {self._format_timestamp(vad_state.first_speech_time) if vad_state.first_speech_time else None}")
            logger.debug(f"  conv_start: {self._format_timestamp(vad_state.conversation_start) if vad_state.conversation_start else None}")
            logger.debug(f"  last_speech: {self._format_timestamp(vad_state.last_speech_time) if vad_state.last_speech_time else None}")
            logger.debug(f"  conv_end: {self._format_timestamp(vad_state.conversation_end) if vad_state.conversation_end else None}")
            
            # First speech detection
            if vad_state.first_speech_time is not None:
                if not self.current_conversation:
                    logger.info(f"Starting new conversation tracking at {self._format_timestamp(vad_state.first_speech_time)} ({vad_state.first_speech_time})")
                    self.current_conversation = {
                        'speech_start': vad_state.first_speech_time,
                        'conversation_start': None,
                        'last_speech': None,
                        'conversation_end': None
                    }
                    timeline_updated = True
            
            # Conversation started
            if vad_state.conversation_start is not None and self.current_conversation:
                if self.current_conversation['conversation_start'] is None:
                    logger.info(f"Marking conversation start at {self._format_timestamp(vad_state.conversation_start)} ({vad_state.conversation_start})")
                    self.current_conversation['conversation_start'] = vad_state.conversation_start
                    timeline_updated = True
            
            # Update last speech time
            if vad_state.last_speech_time is not None and self.current_conversation:
                logger.debug(f"Updating last speech time to {self._format_timestamp(vad_state.last_speech_time)} ({vad_state.last_speech_time})")
                self.current_conversation['last_speech'] = vad_state.last_speech_time
                timeline_updated = True
            
            # Conversation ended
            if vad_state.conversation_end is not None and self.current_conversation:
                if self.current_conversation['conversation_end'] is None:
                    logger.info(f"Completing conversation at {self._format_timestamp(vad_state.conversation_end)} ({vad_state.conversation_end})")
                    self.current_conversation['conversation_end'] = vad_state.conversation_end
                    # Add completed conversation to list
                    self.conversations.append(self.current_conversation)
                    self.current_conversation = None
                    timeline_updated = True
            
            # If timeline was updated or we have an active conversation, queue a redraw
            if timeline_updated or self.current_conversation is not None:
                logger.debug("=== Queueing Timeline Update ===")
                logger.debug(f"Timeline updated: {timeline_updated}")
                logger.debug(f"Active conversation: {self.current_conversation is not None}")
                logger.debug(f"Total conversations: {len(self.conversations)}")
                try:
                    self.update_queue.put_nowait(self._update_conversation_timeline)
                    logger.debug("Successfully queued timeline update")
                except Full:
                    logger.warning("Update queue is full, skipping timeline update")

    def _draw_conversation_state(self, conv: dict, is_current: bool = False) -> None:
        """Draw conversation state visualization."""
        now = time.time()
        alpha_mult = 1.0 if is_current else 0.7
        
        # Create immutable state object
        state = ConversationState.from_dict(conv, now)
        ranges, markers = state.get_ranges(now)
        
        # Draw all ranges
        for color, time_range in ranges:
            self.ax3.axvspan(
                time_range.start,
                time_range.end,
                ymin=0, ymax=1,
                alpha=(0.5 if color == 'yellow' else 0.4 if color == 'orange' else 0.3) * alpha_mult,
                color=color
            )
        
        # Draw all markers
        for color, marker_time, _ in markers:
            self.ax3.axvline(
                x=marker_time,
                color=color,
                linestyle='-',
                linewidth=3 if color == 'purple' else 2,
                alpha=1.0 * alpha_mult
            )

    def _update_conversation_timeline(self):
        """Update the conversation timeline visualization."""
        self.ax3.clear()
        
        # Re-add grid and labels
        self.ax3.grid(True, alpha=0.3)
        self.ax3.set_ylabel('Conversation State')
        self.ax3.set_ylim(-0.1, 1.1)
        
        # Re-add legend
        self.ax3.legend(handles=self.legend_elements, loc='upper right')
        
        # Get time range
        now = time.time()
        
        # Find earliest event time
        earliest_time = now
        if self.conversations:
            for conv in self.conversations:
                if conv['speech_start']:
                    earliest_time = min(earliest_time, conv['speech_start'])
        if self.current_conversation and self.current_conversation['speech_start']:
            earliest_time = min(earliest_time, self.current_conversation['speech_start'])
        
        # Center the timeline around the midpoint between earliest event and now
        window_size = 120  # Show 2 minutes total
        midpoint = (now + earliest_time) / 2
        x_min = midpoint - window_size/2
        x_max = midpoint + window_size/2
        
        # If we're too close to the start, shift window
        if now - x_min < 30:  # Ensure at least 30s future visibility
            x_min = now - (window_size - 30)
            x_max = now + 30
        
        logger.debug(f"Timeline window: {self._format_timestamp(x_min)} to {self._format_timestamp(x_max)}")
        logger.debug(f"Raw timestamps - x_min: {x_min}, x_max: {x_max}, now: {now}")
        self.ax3.set_xlim(x_min, x_max)
        
        # Format timestamps for x-axis (every 30 seconds)
        x_ticks = np.arange(np.floor(x_min), np.ceil(x_max), 30)
        self.ax3.set_xticks(x_ticks)
        tick_labels = [self._format_timestamp(t) for t in x_ticks]
        logger.debug("X-axis ticks:")
        for t, l in zip(x_ticks, tick_labels):
            logger.debug(f"  {t} -> {l}")
        self.ax3.set_xticklabels(tick_labels, rotation=45, ha='right')
        
        # Plot completed conversations
        for conv in self.conversations:
            self._draw_conversation_state(conv)
        
        # Plot current conversation if active
        if self.current_conversation:
            self._draw_conversation_state(self.current_conversation, True)

    def _on_key(self, event):
        """Handle key press events."""
        if event.key == 'q':
            self.running = False
            self.cleanup()
        elif event.key == 'm' and self.audio_player:
            self.audio_player.toggle_mute()
        elif event.key == 'r':
            self.reset_timeline()
    
    def _on_close(self, event):
        """Handle window close event."""
        self.running = False
        self.cleanup()
    
    def cleanup(self):
        """Clean up resources."""
        if not hasattr(self, '_cleanup_called'):
            self._cleanup_called = True
            logger.debug("Cleaning up visualizer")
            
            # Stop animation first
            if self.anim is not None:
                try:
                    self.anim.event_source.stop()
                except Exception as e:
                    logger.error(f"Error stopping animation: {e}")
                self.anim = None
            
            # Disconnect event handlers
            if hasattr(self, '_close_cid') and self.fig.canvas:
                self.fig.canvas.mpl_disconnect(self._close_cid)
            if hasattr(self, '_key_cid') and self.fig.canvas:
                self.fig.canvas.mpl_disconnect(self._key_cid)
            
            # Clear data structures
            self.audio_data.clear()
            self.speech_probs.clear()
            self.is_speech.clear()
            self.conversations.clear()
            self.current_conversation = None
            
            # Clear the update queue
            while not self.update_queue.empty():
                try:
                    self.update_queue.get_nowait()
                except Empty:
                    break
            
            # Close plot windows
            try:
                plt.close(self.fig)
            except Exception as e:
                logger.error(f"Error closing figure: {e}")
            
            # Clear references
            self.fig = None
            self.ax1 = None
            self.ax2 = None
            self.ax3 = None
            self.line_audio = None
            self.line_prob = None
            self.line_speech = None
            self.audio_player = None
    
    def start_animation(self):
        """Start the animation."""
        if self.anim is not None:
            try:
                self.anim.event_source.stop()
            except Exception as e:
                logger.error(f"Error stopping previous animation: {e}")
            self.anim = None
        
        logger.debug("Starting new animation")
        self.anim = FuncAnimation(
            self.fig,
            self._update,
            interval=100,  # Reduced update frequency
            blit=False,    # Disable blitting for more reliable updates
            cache_frame_data=False
        )
        logger.debug("Animation started")

    def set_audio_player(self, player):
        """Set audio player reference for mute toggle."""
        self.audio_player = player
    
    def show(self):
        """Show visualization window."""
        if self.running:
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
        vad = VADProcessor()
        viz = VADVisualizer()
        player = AudioPlayer()
        viz.set_audio_player(player)
        
        # Start audio capture
        def on_data(original_chunk: np.ndarray, downsampled_chunk: np.ndarray):
            """Process captured audio data."""
            try:
                # Add audio to VAD
                vad.add_audio(downsampled_chunk)
                
                # Get VAD state
                state = vad.get_state()
                
                # Update visualization
                viz.add_data(original_chunk, state)
                
                # Play audio
                player.play(original_chunk)
                
            except Exception as e:
                logger.error(f"Error processing audio: {e}")
        
        # Create and start audio capture
        capture = AudioCapture(
            host="staple.local",
            on_data=on_data
        )
        capture.start()
        
        # Start visualization
        viz.start_animation()
        viz.show()
        
    except Exception as e:
        logger.error(f"Error in main: {e}")
    finally:
        logger.info("Cleaning up...")
        # Clean up components
        if capture:
            capture.stop()
        if vad:
            vad.stop()
        if player:
            player.stop()
        if viz:
            viz.cleanup()

if __name__ == "__main__":
    main()

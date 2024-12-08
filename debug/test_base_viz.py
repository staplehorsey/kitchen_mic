"""Test script for base visualization framework."""

import sys
import time
from pathlib import Path
import collections
from typing import Tuple

import numpy as np

# Add src directory to Python path
src_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(src_dir))

from src.visualization.base import BaseVisualizer

class SineWaveVisualizer(BaseVisualizer):
    """Simple sine wave visualization for testing."""
    
    def __init__(self):
        super().__init__(
            num_subplots=2,
            title="Sine Wave Test"
        )
        
        # Data storage
        self.times = collections.deque(maxlen=self.window_size)
        self.sine1 = collections.deque(maxlen=self.window_size)
        self.sine2 = collections.deque(maxlen=self.window_size)
        
        # Initialize with zeros
        now = time.time()
        for i in range(self.window_size):
            self.times.append(now - (self.window_size - i) * 0.02)
            self.sine1.append(0)
            self.sine2.append(0)
        
        # Set up plots
        self.line1, = self.axes[0].plot([], [], 'cyan', label='1 Hz', linewidth=2)
        self.line2, = self.axes[1].plot([], [], 'lime', label='0.5 Hz', linewidth=2)
        
        # Configure axes
        self.configure_axis(
            self.axes[0],
            "Fast Sine Wave",
            "Amplitude",
            (-1.5, 1.5)
        )
        self.configure_axis(
            self.axes[1],
            "Slow Sine Wave",
            "Amplitude",
            (-1.5, 1.5)
        )
    
    def _update(self, frame) -> Tuple:
        with self.data_lock:
            # Add new data points
            now = time.time()
            self.times.append(now)
            self.sine1.append(np.sin(2 * np.pi * 1.0 * now))
            self.sine2.append(np.sin(2 * np.pi * 0.5 * now))
            
            # Update plot data
            times = np.array(self.times)
            sine1 = np.array(self.sine1)
            sine2 = np.array(self.sine2)
            
            # Show last 10 seconds
            window_start = now - 10
            mask = times >= window_start
            plot_times = times[mask] - now  # Relative to current time
            
            self.line1.set_data(plot_times, sine1[mask])
            self.line2.set_data(plot_times, sine2[mask])
            
            # Update axis limits
            for ax in self.axes:
                ax.set_xlim(min(plot_times), max(plot_times))
        
        return self.line1, self.line2

def main():
    """Main test function."""
    try:
        # Create and start visualization
        viz = SineWaveVisualizer()
        viz.start_animation()
        viz.show()
    
    except Exception as e:
        print(f"Error in main: {e}")
    
    finally:
        if viz:
            viz.cleanup()

if __name__ == "__main__":
    main()

"""Audio device configuration and detection for Kitchen Mic."""

import logging
import sounddevice as sd
from typing import Optional, Dict, List, Any
import numpy as np

logger = logging.getLogger(__name__)

class AudioDeviceManager:
    """Manages audio device detection and configuration."""

    @staticmethod
    def list_devices() -> List[Dict[str, Any]]:
        """List all available audio input devices.
        
        Returns:
            List of device information dictionaries.
        """
        devices = []
        try:
            for device in sd.query_devices():
                if device['max_input_channels'] > 0:  # Input devices only
                    devices.append({
                        'name': device['name'],
                        'index': device['index'],
                        'channels': device['max_input_channels'],
                        'default_samplerate': device['default_samplerate'],
                        'supported_samplerates': AudioDeviceManager._get_supported_rates(device['index'])
                    })
        except Exception as e:
            logger.error(f"Error listing audio devices: {e}")
            
        return devices

    @staticmethod
    def _get_supported_rates(device_index: int) -> List[int]:
        """Get supported sample rates for a device.
        
        Args:
            device_index: Index of the audio device.
            
        Returns:
            List of supported sample rates.
        """
        standard_rates = [44100, 48000, 96000, 16000, 22050, 32000]
        supported = []
        
        for rate in standard_rates:
            try:
                sd.check_input_settings(device=device_index, 
                                      samplerate=rate,
                                      channels=1)
                supported.append(rate)
            except Exception:
                continue
                
        return supported

    @staticmethod
    def find_best_device(criteria: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Find the best audio input device based on criteria.
        
        Args:
            criteria: Dictionary of device selection criteria.
            
        Returns:
            Best matching device info or None if no suitable device found.
        """
        devices = AudioDeviceManager.list_devices()
        if not devices:
            return None
            
        # Score each device
        scored_devices = []
        preferred_names = criteria.get('preferred_names', [])
        min_channels = criteria.get('min_channels', 1)
        preferred_rates = criteria.get('preferred_rates', [44100, 48000])
        
        for device in devices:
            if device['channels'] < min_channels:
                continue
                
            score = 0
            
            # Score based on name preferences
            device_name = device['name'].lower()
            for i, name in enumerate(preferred_names):
                if name.lower() in device_name:
                    score += len(preferred_names) - i
                    
            # Score based on sample rate support
            supported_rates = set(device['supported_samplerates'])
            for i, rate in enumerate(preferred_rates):
                if rate in supported_rates:
                    score += len(preferred_rates) - i
                    
            scored_devices.append((score, device))
            
        if not scored_devices:
            return None
            
        # Return device with highest score
        return max(scored_devices, key=lambda x: x[0])[1]

    @staticmethod
    def validate_device_config(device: Optional[Dict[str, Any]], config: Dict[str, Any]) -> bool:
        """Validate device against required configuration.
        
        Args:
            device: Device information dictionary.
            config: Audio configuration dictionary.
            
        Returns:
            True if configuration is valid, False otherwise.
        """
        if not device:
            return False
            
        try:
            # Check if device supports required sample rates
            capture_rate = config['capture']['sample_rate']
            vad_rate = config['vad']['sample_rate']
            
            supported_rates = set(device['supported_samplerates'])
            if capture_rate not in supported_rates:
                logger.error(f"Device does not support capture rate {capture_rate}")
                return False
                
            if vad_rate not in supported_rates:
                logger.error(f"Device does not support VAD rate {vad_rate}")
                return False
                
            # Validate other settings
            sd.check_input_settings(
                device=device['index'],
                samplerate=capture_rate,
                channels=config['capture']['channels']
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Device validation failed: {e}")
            return False

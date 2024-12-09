#!/usr/bin/env python3
"""Configuration management for Kitchen Mic."""

import os
import sys
import yaml
import logging
import platform
import socket
import requests
from pathlib import Path
from typing import Optional, Dict, Any, List
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

class SystemInfo:
    """System information and paths following Unix conventions."""
    
    @staticmethod
    def get_system() -> str:
        """Get the current operating system."""
        system = platform.system().lower()
        if system == "darwin":
            return "macos"
        return system

    @staticmethod
    def get_config_paths() -> List[Path]:
        """Get configuration file search paths following XDG spec.
        
        Returns:
            List of paths to search for config files, in order of precedence.
        """
        paths = []
        
        # 1. Check XDG_CONFIG_HOME or ~/.config
        xdg_config = os.environ.get('XDG_CONFIG_HOME', 
                                  os.path.expanduser('~/.config'))
        paths.append(Path(xdg_config) / 'kitchen_mic')
        
        # 2. Check XDG_CONFIG_DIRS or /etc/xdg
        xdg_config_dirs = os.environ.get('XDG_CONFIG_DIRS', '/etc/xdg')
        for config_dir in xdg_config_dirs.split(':'):
            if config_dir:
                paths.append(Path(config_dir) / 'kitchen_mic')
        
        # 3. Add system-wide config location
        paths.append(Path('/etc/kitchen_mic'))
        
        return paths

    @staticmethod
    def get_data_paths() -> List[Path]:
        """Get data directory paths following XDG spec.
        
        Returns:
            List of paths to search for data, in order of precedence.
        """
        paths = []
        
        # 1. Check XDG_DATA_HOME or ~/.local/share
        xdg_data = os.environ.get('XDG_DATA_HOME',
                                os.path.expanduser('~/.local/share'))
        paths.append(Path(xdg_data) / 'kitchen_mic')
        
        # 2. Check XDG_DATA_DIRS or /usr/local/share/:/usr/share/
        xdg_data_dirs = os.environ.get('XDG_DATA_DIRS',
                                     '/usr/local/share/:/usr/share/')
        for data_dir in xdg_data_dirs.split(':'):
            if data_dir:
                paths.append(Path(data_dir) / 'kitchen_mic')
                
        return paths

    @staticmethod
    def get_removable_mount_points() -> List[Path]:
        """Get system-specific removable drive mount points.
        
        Returns:
            List of paths where removable drives might be mounted.
        """
        system = SystemInfo.get_system()
        
        if system == "macos":
            return [Path("/Volumes")]
        elif system == "linux":
            # Common Linux mount points for removable media
            return [
                Path("/media"),
                Path("/mnt"),
                Path(os.path.expanduser("~")) / "media"
            ]
        return []

    @staticmethod
    def get_default_storage_path() -> Path:
        """Get default storage path following XDG spec.
        
        Returns:
            Default path for storing application data.
        """
        xdg_data = os.environ.get('XDG_DATA_HOME',
                                os.path.expanduser('~/.local/share'))
        return Path(xdg_data) / 'kitchen_mic' / 'storage'


class ConfigManager:
    """Manages Kitchen Mic configuration."""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize config manager.
        
        Args:
            config_path: Path to config file. If None, searches standard locations.
        """
        self.system = SystemInfo.get_system()
        self.config_path = self._find_config(config_path) if config_path else None
        self.config: Dict[str, Any] = {}
        self._load_config()
        self._validate_config()

    def _find_config(self, config_path: Optional[str] = None) -> Optional[Path]:
        """Find configuration file in standard locations.
        
        Args:
            config_path: Explicit path to config file, if provided.
            
        Returns:
            Path to found config file, or None if not found.
        """
        if config_path:
            path = Path(config_path)
            return path if path.is_file() else None
            
        # Search standard locations
        for path in SystemInfo.get_config_paths():
            config_file = path / "config.yaml"
            if config_file.is_file():
                return config_file
                
        # Fall back to package default
        default_config = Path(__file__).parent.parent.parent / "config" / "default.yaml"
        return default_config if default_config.is_file() else None

    def _load_config(self) -> None:
        """Load configuration from YAML file."""
        if not self.config_path:
            raise FileNotFoundError("No configuration file found")
            
        try:
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {self.config_path}")
        except Exception as e:
            logger.error(f"Failed to load config from {self.config_path}: {e}")
            raise

    def _validate_config(self) -> None:
        """Validate configuration."""
        # Validate required sections
        required = ['storage', 'audio', 'vad', 'models', 'service']
        if not all(k in self.config for k in required):
            raise ValueError("Missing required configuration sections")
        
        # Validate each section
        self._validate_storage_config()
        self._validate_audio_source()
        self._validate_llm_endpoint()

    def _validate_storage_config(self) -> None:
        """Validate storage configuration."""
        storage_config = self.config.get('storage', {})
        if 'base_dir' not in storage_config:
            raise ValueError("Missing required 'base_dir' in storage config")
            
        base_path = Path(storage_config['base_dir'])
        if not base_path.exists():
            logging.error(f"Storage directory {base_path} does not exist")
            raise ValueError(f"Storage directory {base_path} does not exist. Please run install.sh to set it up.")

        # Validate directory is writable
        test_file = base_path / '.write_test'
        try:
            test_file.touch()
            test_file.unlink()
        except Exception as e:
            logging.error(f"Storage directory {base_path} is not writable: {e}")
            raise ValueError(f"Storage directory {base_path} is not writable. Please check permissions.")

        logging.info(f"Storage directory {base_path} is valid and writable")

    def _validate_audio_source(self) -> None:
        """Validate audio source configuration."""
        audio_config = self.config.get('audio', {})
        source = audio_config.get('source', {})
        
        # Validate required fields
        required = ['type', 'buffer_size']
        if not all(k in source for k in required):
            raise ValueError("Missing required audio source configuration")
            
        # Validate source type
        if source['type'] != 'microphone':
            raise ValueError(f"Invalid audio source type: {source['type']}")
            
        # Validate audio capture settings
        capture = audio_config.get('capture', {})
        required_capture = ['sample_rate', 'channels', 'chunk_size']
        if not all(k in capture for k in required_capture):
            raise ValueError("Missing required audio capture configuration")
            
        # Validate sample rates and chunk sizes
        if not (8000 <= capture['sample_rate'] <= 192000):
            raise ValueError(f"Invalid sample rate: {capture['sample_rate']}")
            
        if not (1 <= capture['channels'] <= 2):
            raise ValueError(f"Invalid channel count: {capture['channels']}")
            
        if not (64 <= capture['chunk_size'] <= 8192):
            raise ValueError(f"Invalid chunk size: {capture['chunk_size']}")
            
        logger.info("Audio configuration validated successfully")

    def _validate_llm_endpoint(self) -> None:
        """Validate LLM endpoint configuration."""
        llm_config = self.config['models']['llm']
        
        # Validate endpoint URL
        endpoint = llm_config['endpoint']
        try:
            parsed = urlparse(endpoint)
            if not all([parsed.scheme, parsed.netloc]):
                raise ValueError(f"Invalid LLM endpoint URL: {endpoint}")
        except Exception as e:
            raise ValueError(f"Invalid LLM endpoint URL: {e}")
            
        # Validate other parameters
        if not (0 <= llm_config['temperature'] <= 1):
            raise ValueError(f"Invalid temperature: {llm_config['temperature']}")
            
        if llm_config['max_tokens'] < 1:
            raise ValueError(f"Invalid max_tokens: {llm_config['max_tokens']}")
            
        # Test endpoint if possible
        try:
            response = requests.get(
                f"{endpoint}/health",
                timeout=llm_config['timeout_sec']
            )
            response.raise_for_status()
            logger.info(f"Successfully connected to LLM endpoint at {endpoint}")
        except Exception as e:
            logger.warning(f"Could not connect to LLM endpoint: {e}")
            # Don't raise error as the service might not be running during config validation

    def get_config(self) -> Dict[str, Any]:
        """Get the complete configuration dictionary."""
        return self.config

    def get_storage_path(self) -> Path:
        """Get the resolved storage path."""
        return Path(self.config['storage']['base_dir'])

    def get_audio_config(self) -> Dict[str, Any]:
        """Get audio configuration."""
        return self.config['audio']

    def get_vad_config(self) -> Dict[str, Any]:
        """Get VAD configuration."""
        return self.config['vad']

    def get_audio_device(self) -> Optional[Dict[str, Any]]:
        """Get the configured audio device information.
        
        Returns:
            Dictionary with device information or None if using system default.
        """
        devices = AudioDeviceManager.list_devices()
        device_index = self.config['audio']['input_device']
        
        if not device_index:
            return None
            
        return next(
            (d for d in devices if d['index'] == device_index),
            None
        )

    def _find_usb_storage(self) -> Optional[Path]:
        """Find suitable USB storage device.
        
        Returns:
            Path to USB storage if found, None otherwise.
        """
        min_space_gb = self.config['storage']['min_free_space_gb']
        
        for mount_point in SystemInfo.get_removable_mount_points():
            if not mount_point.exists():
                continue
                
            # Check each mounted device
            for volume in mount_point.iterdir():
                if not volume.is_dir():
                    continue
                    
                try:
                    stats = os.statvfs(volume)
                    free_gb = (stats.f_bavail * stats.f_frsize) / (1024**3)
                    
                    if free_gb >= min_space_gb:
                        target_dir = volume / "kitchen_mic_data"
                        target_dir.mkdir(exist_ok=True)
                        return target_dir
                        
                except Exception as e:
                    logger.debug(f"Skipping volume {volume}: {e}")
                    continue
                    
        return None

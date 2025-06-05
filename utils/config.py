import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional



class ConfigManager:
    """Configuration manager for loading, accessing, and updating YAML configuration files."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = Path(config_path)
        self._config = None
        self.load_config()
    
    # Load configuration from the specified YAML file
    def load_config(self):
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                self._config = yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML config: {e}")
    
    # Access config values using dot notation
    def get(self, key: str, default: Any = None) -> Any:
        """Get config value using dot notation (e.g., 'models.mobileclip.model_name')"""
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value
    
    # Get entire config section
    def get_section(self, section: str) -> Dict[str, Any]:
        """Get entire config section"""
        return self._config.get(section, {})
    
    # Update config values using dot notation
    def update(self, key: str, value: Any):
        """Update config value and save to file"""
        keys = key.split('.')
        config = self._config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
        self.save_config()
    
    # Save current config to file
    def save_config(self, path: Optional[str] = None):
        """Save current config to file"""
        save_path = path or self.config_path
        with open(save_path, 'w') as f:
            yaml.dump(self._config, f, default_flow_style=False, indent=2)
    
    # Property to access the full config dictionary
    @property
    def config(self) -> Dict[str, Any]:
        """Get full config dictionary"""
        return self._config


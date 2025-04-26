"""
Configuration manager for RTD simulation parameters.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from ..model.rtd_model import RTDModel

class ConfigManager:
    """Manages RTD simulation configuration files."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Path to configuration file. If None, uses default config.
        """
        self.config_path = config_path
        self.config: Dict[str, Any] = {}
        
    def load_config(self) -> Dict[str, Any]:
        """
        Load configuration from file or use defaults.
        
        Returns:
            Dictionary containing configuration parameters.
        """
        if self.config_path and Path(self.config_path).exists():
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            # Default configuration
            self.config = {
                'model': {
                    'm': 0.078,
                    'initial_v': -1.1,
                    'initial_i': None,  # Will be calculated from IV function
                    'dt': 0.01,
                    't_end': 100.0
                },
                'simulation': {
                    'vbias': 0.0,
                    'perturbation_type': 'none',
                    'perturbation_params': {}
                }
            }
        return self.config
    
    def save_config(self, config: Dict[str, Any], path: Optional[str] = None) -> None:
        """
        Save configuration to file.
        
        Args:
            config: Configuration dictionary to save
            path: Path to save configuration file. If None, uses self.config_path
        """
        save_path = path or self.config_path
        if not save_path:
            raise ValueError("No path specified for saving configuration")
            
        with open(save_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    
    def create_rtd_model(self) -> RTDModel:
        """
        Create an RTD model instance from the current configuration.
        
        Returns:
            RTDModel instance configured with current parameters
        """
        if not self.config:
            self.load_config()
            
        model_config = self.config['model']
        return RTDModel(
            m=model_config['m'],
            initial_v=model_config['initial_v'],
            initial_i=model_config['initial_i']
        )
    
    def update_from_model(self, model: RTDModel) -> None:
        """
        Update configuration from an RTD model instance.
        
        Args:
            model: RTDModel instance to get parameters from
        """
        if not self.config:
            self.load_config()
            
        self.config['model'].update({
            'm': model.m,
            'initial_v': model.v,
            'initial_i': model.i
        }) 
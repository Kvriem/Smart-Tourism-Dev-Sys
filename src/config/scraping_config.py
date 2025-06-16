"""
Configuration settings for hotel scraping DAG
"""
import json
import os
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional

# Scraping limits for quarterly runs
QUARTERLY_SCRAPING_CONFIG = {
    'max_hotels_per_run': 50,
    'max_pages_per_hotel': 5,
    'batch_size': 5,  # Number of hotels before driver recreation
    'max_retries_per_hotel': 3,
}

# Database configuration
DATABASE_CONFIG = {
    'connection_timeout': 30,
    'query_timeout': 300,
    'batch_insert_size': 100,
}

# Driver configuration
DRIVER_CONFIG = {
    'implicit_wait': 5,
    'page_load_timeout': 30,
    'element_wait_timeout': 15,
}

# Delay configuration
DELAY_CONFIG = {
    'base_delay': 3,
    'variance': 2,
    'page_delay': 4,
    'hotel_delay': 8,
}

# Quarterly schedule settings
SCHEDULE_CONFIG = {
    'timezone': 'UTC',
    'hour': 2,  # 2 AM UTC
    'quarters': {
        'Q1': {'month': 3, 'target_quarter': 'Dec-Feb'},
        'Q2': {'month': 6, 'target_quarter': 'Mar-May'},
        'Q3': {'month': 9, 'target_quarter': 'Jun-Aug'},
        'Q4': {'month': 12, 'target_quarter': 'Sep-Nov'},
    }
}

# Preset configurations
PRESET_CONFIGS = {
    'development': {
        'max_hotels_per_run': 5,
        'max_pages_per_hotel': 2,
        'base_delay_between_requests': 1,
        'delay_between_hotels': 2,
        'headless_mode': False,
        'test_mode': True
    },
    'testing': {
        'max_hotels_per_run': 10,
        'max_pages_per_hotel': 3,
        'base_delay_between_requests': 2,
        'delay_between_hotels': 3,
        'headless_mode': True,
        'test_mode': True
    },
    'production': {
        'max_hotels_per_run': 50,
        'max_pages_per_hotel': 5,
        'base_delay_between_requests': 3,
        'delay_between_hotels': 8,
        'headless_mode': True,
        'test_mode': False
    }
}

@dataclass
class ScrapingConfig:
    """Configuration data class for scraping parameters"""
    max_hotels_per_run: int = 50
    max_pages_per_hotel: int = 5
    base_delay_between_requests: int = 3
    delay_between_hotels: int = 8
    headless_mode: bool = True
    test_mode: bool = False
    target_quarter: Optional[str] = None
    batch_size: int = 5
    max_retries_per_hotel: int = 3

class ConfigManager:
    """Manages scraping configuration"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file
        self.config = ScrapingConfig()
        if config_file and os.path.exists(config_file):
            self._load_config()
    
    def _load_config(self):
        """Load configuration from file"""
        try:
            with open(self.config_file, 'r') as f:
                config_data = json.load(f)
                for key, value in config_data.items():
                    if hasattr(self.config, key):
                        setattr(self.config, key, value)
        except Exception as e:
            print(f"Warning: Could not load config file {self.config_file}: {e}")
    
    def save_config(self):
        """Save current configuration to file"""
        if self.config_file:
            try:
                with open(self.config_file, 'w') as f:
                    json.dump(asdict(self.config), f, indent=2)
            except Exception as e:
                print(f"Warning: Could not save config file {self.config_file}: {e}")
    
    def update_config(self, **kwargs):
        """Update configuration parameters"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
    
    def get_config_dict(self) -> Dict[str, Any]:
        """Get configuration as dictionary"""
        return asdict(self.config)

# Global config manager instance
config_manager = ConfigManager()

def get_runtime_config() -> Dict[str, Any]:
    """Get runtime configuration as dictionary"""
    return config_manager.get_config_dict()

def load_preset(preset_name: str) -> bool:
    """Load a predefined configuration preset"""
    if preset_name in PRESET_CONFIGS:
        config_manager.update_config(**PRESET_CONFIGS[preset_name])
        return True
    return False

def create_config_template(file_path: str):
    """Create a configuration template file"""
    template_config = asdict(ScrapingConfig())
    try:
        with open(file_path, 'w') as f:
            json.dump(template_config, f, indent=2)
        print(f"Configuration template created at: {file_path}")
    except Exception as e:
        print(f"Error creating config template: {e}")

def get_quarterly_config():
    """Get quarterly-specific configuration"""
    return QUARTERLY_SCRAPING_CONFIG

def get_database_config():
    """Get database configuration"""
    return DATABASE_CONFIG

def get_driver_config():
    """Get driver configuration"""
    return DRIVER_CONFIG

def get_delay_config():
    """Get delay configuration"""
    return DELAY_CONFIG

def get_schedule_config():
    """Get schedule configuration"""
    return SCHEDULE_CONFIG

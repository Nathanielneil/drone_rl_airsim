#!/usr/bin/env python3
"""
Unit tests for configuration management
"""
import unittest
import tempfile
import os
import yaml
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from src.core.config_manager import ConfigManager


class TestConfigManager(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        self.config_manager = ConfigManager()
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test config files
        self.test_config = {
            "algorithm": "sac",
            "training": {
                "total_timesteps": 100000,
                "batch_size": 256
            },
            "env": {
                "name": "test_env"
            }
        }
        
        self.config_file = os.path.join(self.temp_dir, "test_config.yaml")
        with open(self.config_file, 'w') as f:
            yaml.dump(self.test_config, f)
    
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_load_config(self):
        """Test loading configuration from file"""
        config = self.config_manager.load_config(self.config_file)
        
        self.assertEqual(config["algorithm"], "sac")
        self.assertEqual(config["training"]["total_timesteps"], 100000)
        self.assertEqual(config["env"]["name"], "test_env")
    
    def test_merge_configs(self):
        """Test merging multiple configurations"""
        base_config = {
            "algorithm": "ppo",
            "training": {
                "total_timesteps": 50000,
                "batch_size": 128
            }
        }
        
        override_config = {
            "training": {
                "total_timesteps": 200000
            },
            "env": {
                "max_steps": 1000
            }
        }
        
        merged = self.config_manager.merge_configs(base_config, override_config)
        
        # Should keep original algorithm
        self.assertEqual(merged["algorithm"], "ppo")
        # Should override timesteps
        self.assertEqual(merged["training"]["total_timesteps"], 200000)
        # Should keep original batch_size
        self.assertEqual(merged["training"]["batch_size"], 128)
        # Should add new env setting
        self.assertEqual(merged["env"]["max_steps"], 1000)
    
    def test_validate_config(self):
        """Test configuration validation"""
        valid_config = {
            "algorithm": "sac",
            "training": {
                "total_timesteps": 100000
            }
        }
        
        invalid_config = {
            "algorithm": "invalid_algo",
            "training": {
                "total_timesteps": -1000
            }
        }
        
        # This should not raise an exception
        self.config_manager.validate_config(valid_config)
        
        # This should raise an exception
        with self.assertRaises(ValueError):
            self.config_manager.validate_config(invalid_config)
    
    def test_load_nonexistent_file(self):
        """Test loading non-existent config file"""
        with self.assertRaises(FileNotFoundError):
            self.config_manager.load_config("nonexistent_file.yaml")


if __name__ == "__main__":
    unittest.main()
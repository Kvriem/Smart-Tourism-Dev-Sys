#!/usr/bin/env python3
"""
Manual Hotel Reviews Scraper
============================

This script allows manual execution of the scraping process with configurable parameters.
Use this for testing, development, or manual runs outside of Airflow.
"""

import sys
import argparse
import logging

# Add the src directory to Python path
sys.path.append('/home/kariem/airflow/src')

from config.scraping_config import (
    config_manager, 
    get_runtime_config, 
    load_preset,
    create_config_template,
    PRESET_CONFIGS
)
from scrapers.hotel_scraper import main_scraping_process

def setup_argument_parser():
    """Setup command line argument parser"""
    parser = argparse.ArgumentParser(description="Manual Hotel Reviews Scraper")
    
    # Configuration options
    parser.add_argument('--preset', choices=list(PRESET_CONFIGS.keys()),
                       help='Use a predefined configuration preset')
    parser.add_argument('--hotels', type=int, 
                       help='Maximum number of hotels to process')
    parser.add_argument('--pages', type=int,
                       help='Maximum pages per hotel')
    parser.add_argument('--test', action='store_true',
                       help='Run in test mode (fast, limited)')
    parser.add_argument('--config-file', type=str,
                       help='Path to custom configuration JSON file')
    
    # Specific overrides
    parser.add_argument('--delay', type=float,
                       help='Base delay between requests (seconds)')
    parser.add_argument('--hotel-delay', type=float,
                       help='Delay between hotels (seconds)')
    parser.add_argument('--headless', action='store_true',
                       help='Run browser in headless mode')
    parser.add_argument('--show-browser', action='store_true',
                       help='Show browser (opposite of headless)')
    parser.add_argument('--quarter', type=str,
                       help='Target quarter (e.g., "Mar-May")')
    
    # Utility commands
    parser.add_argument('--create-template', type=str,
                       help='Create configuration template at specified path')
    parser.add_argument('--show-config', action='store_true',
                       help='Show current configuration and exit')
    parser.add_argument('--list-presets', action='store_true',
                       help='List available configuration presets')
    
    return parser

def apply_cli_overrides(args):
    """Apply command line argument overrides to configuration"""
    if args.hotels is not None:
        config_manager.update_config(max_hotels_per_run=args.hotels)
    
    if args.pages is not None:
        config_manager.update_config(max_pages_per_hotel=args.pages)
    
    if args.delay is not None:
        config_manager.update_config(base_delay_between_requests=args.delay)
    
    if args.hotel_delay is not None:
        config_manager.update_config(delay_between_hotels=args.hotel_delay)
    
    if args.headless:
        config_manager.update_config(headless_mode=True)
    
    if args.show_browser:
        config_manager.update_config(headless_mode=False)
    
    if args.quarter:
        config_manager.update_config(target_quarter=args.quarter)
    
    if args.test:
        config_manager.update_config(test_mode=True)

def show_current_config():
    """Display current configuration"""
    config = config_manager.config
    
    print("\n=== Current Scraping Configuration ===")
    print(f"Max Hotels: {config.max_hotels_per_run}")
    print(f"Max Pages per Hotel: {config.max_pages_per_hotel}")
    print(f"Test Mode: {config.test_mode}")
    print(f"Base Delay: {config.base_delay_between_requests}s")
    print(f"Hotel Delay: {config.delay_between_hotels}s")
    print(f"Headless Mode: {config.headless_mode}")
    print(f"Target Quarter: {config.target_quarter}")
    print(f"Driver Recreation: Every {config.driver_recreation_interval} hotels")
    
    print("\n=== Runtime Configurations ===")
    for run_type in ["normal", "quarterly", "test"]:
        runtime_config = get_runtime_config(run_type)
        print(f"{run_type.upper()}: {runtime_config}")

def list_presets():
    """List available configuration presets"""
    print("\n=== Available Configuration Presets ===")
    for name, config in PRESET_CONFIGS.items():
        print(f"\n{name.upper()}:")
        print(f"  Max Hotels: {config.max_hotels_per_run}")
        print(f"  Max Pages: {config.max_pages_per_hotel}")
        print(f"  Test Mode: {config.test_mode}")
        print(f"  Base Delay: {config.base_delay_between_requests}s")
        print(f"  Headless: {config.headless_mode}")

def main():
    """Main function for manual scraper"""
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    # Handle utility commands first
    if args.create_template:
        create_config_template(args.create_template)
        print(f"Configuration template created at: {args.create_template}")
        return
    
    if args.list_presets:
        list_presets()
        return
    
    if args.show_config:
        show_current_config()
        return
    
    # Load configuration file if specified
    if args.config_file:
        config_manager.config_file = args.config_file
        config_manager._load_config()
        print(f"Loaded configuration from: {args.config_file}")
    
    # Load preset if specified
    if args.preset:
        load_preset(args.preset)
        print(f"Loaded preset configuration: {args.preset}")
    
    # Apply CLI overrides
    apply_cli_overrides(args)
    
    # Show final configuration
    print("\n=== Starting Manual Scraping ===")
    show_current_config()
    
    # Get runtime configuration
    run_type = "test" if args.test or config_manager.config.test_mode else "normal"
    runtime_config = get_runtime_config(run_type)
    
    print(f"\nUsing runtime configuration: {runtime_config}")
    
    # Start scraping
    try:
        main_scraping_process(
            max_hotels=runtime_config["max_hotels"],
            max_pages_per_hotel=runtime_config["max_pages_per_hotel"]
        )
        print("\n✅ Scraping completed successfully!")
    except KeyboardInterrupt:
        print("\n⚠️ Scraping interrupted by user")
    except Exception as e:
        print(f"\n❌ Scraping failed: {e}")
        logging.error(f"Manual scraping failed: {e}")

if __name__ == "__main__":
    main()

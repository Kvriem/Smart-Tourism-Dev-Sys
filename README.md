# Hotel Reviews Scraping and NLP Pipeline

## Project Structure

```
/home/kariem/airflow/
├── dags/                          # Airflow DAGs (detected by Airflow)
│   └── scheduled.py               # Main quarterly scraping DAG
├── src/                           # Source code modules
│   ├── scrapers/                  # Web scraping modules
│   │   └── hotel_scraper.py       # Main hotel scraping logic
│   ├── nlp/                       # NLP processing modules
│   │   ├── NLP_Script.py          # NLP processing script
│   │   └── preprocessing_nlp_ref/ # NLP preprocessing reference
│   ├── config/                    # Configuration modules
│   │   ├── scraping_config.py     # Scraping configuration
│   │   └── webserver_config.py    # Airflow webserver config
│   └── utils/                     # Utility modules
│       └── drivers/               # WebDriver binaries
├── scripts/                       # Standalone scripts
│   ├── manual_scraper.py          # Manual scraping script
│   ├── test_firefox.py            # Firefox testing
│   ├── simple_firefox_test.py     # Simple Firefox test
│   ├── fix_airflow_installation.sh
│   └── install_firefox.sh
├── data/                          # Data storage (empty)
├── tests/                         # Test files (empty)
├── logs/                          # Airflow logs
├── config/                        # Airflow configuration
├── requirements.txt               # Python dependencies
├── airflow.cfg                    # Airflow configuration
└── airflow.db                     # Airflow SQLite database
```

## Quick Start

1. **Activate Virtual Environment:**
   ```bash
   cd /home/kariem/airflow
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Start Airflow:**
   ```bash
   airflow standalone
   ```

3. **Run Manual Scraping:**
   ```bash
   cd scripts
   python manual_scraper.py --preset development --hotels 5
   ```

## Key Components

- **scheduled.py**: Main Airflow DAG for quarterly hotel reviews scraping
- **hotel_scraper.py**: Core scraping functionality
- **scraping_config.py**: Configuration management with presets
- **manual_scraper.py**: Manual execution script for testing

## Configuration Presets

- `development`: 5 hotels, 2 pages, no headless mode
- `testing`: 10 hotels, 3 pages, headless mode
- `production`: 50 hotels, 5 pages, headless mode

## Recent Changes

- Restructured project for better organization
- Updated all import paths to use new structure
- Created proper Python packages with `__init__.py` files
- Moved DAGs to proper Airflow location
- Added comprehensive configuration management

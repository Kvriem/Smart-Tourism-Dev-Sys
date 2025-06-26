# Smart Tourism Development System

A comprehensive data pipeline system for tourism data processing, designed for scalable hotel review scraping, processing, and analysis.

## Overview

This project implements an automated data pipeline using Apache Airflow for processing hotel review data. The system is designed to handle large-scale tourism data with efficient processing and translation capabilities.

## Project Structure

```
├── dags/
│   └── scheduled.py          # Main Airflow DAG for data pipeline
└── .gitignore               # Git ignore configuration
```

## Features

- **Automated Data Pipeline**: Scheduled Airflow DAG for continuous data processing
- **Hotel Review Processing**: Comprehensive hotel review scraping and analysis
- **Multi-language Support**: Translation capabilities for global tourism data
- **Scalable Architecture**: Designed for high-volume data processing

## Getting Started

### Prerequisites

- Python 3.8+
- Apache Airflow
- Required Python packages (see requirements.txt)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Kvriem/Smart-Tourism-Dev-Sys.git
cd Smart-Tourism-Dev-Sys
```

2. Set up Apache Airflow environment
3. Place the DAG file in your Airflow dags folder
4. Configure your Airflow connections and variables as needed

## Usage

The main data pipeline is implemented in `dags/scheduled.py`. This DAG handles:

- Data scraping from hotel review sources
- Data preprocessing and cleaning
- Translation processing
- Database ingestion

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the MIT License.

## Contact

For questions and support, please open an issue in this repository.

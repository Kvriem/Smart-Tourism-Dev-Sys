<<<<<<< HEAD
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

﻿## Smart Tourism Development System (Not finished Yet)

### Project Overview

The Smart Tourism Development System is a data-driven solution designed to improve Egypt's tourism sector by leveraging advanced analytics and machine learning. It analyzes tourist feedback to provide actionable insights for stakeholders, fostering enhanced tourist experiences and competitive advantage.

### Key Features

- **Automated Data Collection:** Continuous scraping of tourist feedback for up-to-date insights.
- **Sentiment Analysis:** Categorizes feedback to measure satisfaction levels.
- **Pain Point Identification:** Highlights recurring issues and positive aspects of tourist experiences.
- **Development Plan Suggestions:** Prioritizes improvement recommendations.
- **Role-Based Dashboards:** Offers tailored interfaces for government authorities and businesses.
- **Exportable Reports:** Facilitates sharing of actionable insights.
- **Competitor Analysis:** Benchmarks Egypt against other destinations to highlight opportunities.
- **Predictive Analytics:** Anticipates potential challenges and opportunities using historical data.

### Target Beneficiaries

- **Government and Tourism Authorities:** Leverage data insights for strategic planning and service enhancements.
- **Local Tourism Operators:** Improve service offerings based on real-time feedback and trends.

### Sustainable Development Contributions

- **Economic Growth:** Enhances the tourism sector, increases revenue, and creates jobs.
- **Cultural Preservation:** Promotes the maintenance of Egypt's historical and cultural sites.

### Tech Stack

#### Backend

- **Database:** MongoDB
- **Programming Language:** Python
- **Cloud Services:** AWS (RDS, EC2, Lambda)

#### Frontend

- **Framework:** Angular

#### Analytics and Machine Learning

- **Libraries:** Pandas, NLTK
- **Functions:** Sentiment analysis, data visualization, and predictive modeling.

#### Deployment Tools

- **Hosting:** AWS Elastic Beanstalk

### Project Roadmap

| Phase   | Version | Tasks                                    | Status      |
|---------|---------|------------------------------------------|-------------|
| Phase 1 | v0.1    | Initial project setup                    |   Finshed   |
| Phase 2 | v0.2    | Local data scraping                      | In Progress |
| Phase 3 | v0.3    | Data cleaning and NLP sentiment analysis | In Progress |
| Phase 4 | v1.0    | Database deployment on AWS RDS           | Not Started |
| Phase 5 | v1.2    | Cloud-based data processing              | Not Started |
| Phase 6 | v2.0    | Full web application deployment          | Not Started |


### Team Members

- Kariem Abdelmoniem Ahmed
- Mohamed Ashraf Mohamed
- Kirolos Raouf Helmy
- Ahmed Mohamed Nabil
- Abdelrahman Mohamed Abdelnaby 
- Mahmoud Mohamed Sharfy

### Supervisor

- Dr. Mohamed Fouad


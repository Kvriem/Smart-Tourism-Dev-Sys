import os
import csv
import random
import logging
import datetime
from time import sleep
import pandas as pd
import requests
from bs4 import BeautifulSoup as bs
from requests.exceptions import RequestException
import psycopg2
from psycopg2 import OperationalError

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.firefox import GeckoDriverManager


def initialize_logging():
    """
    Initialize logging configuration with a log file in the script's directory.
    """
    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_file_path = os.path.join(script_dir, 'scraping_log.log')
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file_path),
            logging.StreamHandler()  # Also log to console
        ]
    )
    
    logging.info(f"Logging initialized. Log file: {log_file_path}")


def create_db_connection():
    """
    Create and return a PostgreSQL database connection.
    
    Returns:
        psycopg2.connection: Database connection object or None if connection fails
    """
    connection_string = "postgresql://neondb_owner:npg_ExFXHY8yiNT0@ep-lingering-term-ab7pbfql-pooler.eu-west-2.aws.neon.tech/neondb?sslmode=require"
    
    try:
        connection = psycopg2.connect(connection_string)
        logging.info("Database connection established successfully")
        return connection
    except OperationalError as e:
        logging.error(f"Failed to connect to database: {e}")
        return None


def get_random_user_agent():
    """
    Get a random user agent from a predefined list for better rotation.
    
    Returns:
        str: Random user agent string
    """
    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/121.0",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/120.0",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/121.0"
    ]
    
    return random.choice(user_agents)


def download_geckodriver_manually():
    """
    Download geckodriver manually if webdriver_manager fails.
    
    Returns:
        str: Path to the downloaded geckodriver executable
    """
    import platform
    import zipfile
    import urllib.request
    
    try:
        # Use the utils/drivers directory in the project structure
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        drivers_dir = os.path.join(project_root, 'src', 'utils', 'drivers')
        os.makedirs(drivers_dir, exist_ok=True)
        
        system = platform.system().lower()
        architecture = platform.machine().lower()
        
        # Determine the correct geckodriver URL
        if system == 'windows':
            if '64' in architecture:
                driver_url = "https://github.com/mozilla/geckodriver/releases/download/v0.34.0/geckodriver-v0.34.0-win64.zip"
                driver_filename = "geckodriver.exe"
            else:
                driver_url = "https://github.com/mozilla/geckodriver/releases/download/v0.34.0/geckodriver-v0.34.0-win32.zip"
                driver_filename = "geckodriver.exe"
        elif system == 'darwin':  # macOS
            driver_url = "https://github.com/mozilla/geckodriver/releases/download/v0.34.0/geckodriver-v0.34.0-macos.tar.gz"
            driver_filename = "geckodriver"
        else:  # Linux
            driver_url = "https://github.com/mozilla/geckodriver/releases/download/v0.34.0/geckodriver-v0.34.0-linux64.tar.gz"
            driver_filename = "geckodriver"
        
        driver_path = os.path.join(drivers_dir, driver_filename)
        
        # Check if driver already exists
        if os.path.exists(driver_path):
            logging.info(f"Using existing geckodriver at: {driver_path}")
            return driver_path
        
        logging.info(f"Downloading geckodriver from: {driver_url}")
        
        # Download the driver
        zip_path = os.path.join(drivers_dir, "geckodriver.zip")
        urllib.request.urlretrieve(driver_url, zip_path)
        
        # Extract the driver
        if driver_url.endswith('.zip'):
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(drivers_dir)
        else:  # tar.gz
            import tarfile
            with tarfile.open(zip_path, 'r:gz') as tar_ref:
                tar_ref.extractall(drivers_dir)
        
        # Make executable on Unix systems
        if system != 'windows':
            os.chmod(driver_path, 0o755)
        
        # Clean up zip file
        os.remove(zip_path)
        
        logging.info(f"Successfully downloaded geckodriver to: {driver_path}")
        return driver_path
        
    except Exception as e:
        logging.error(f"Failed to download geckodriver manually: {e}")
        return None


def setup_driver():
    """
    Set up and return a Firefox WebDriver with anti-detection and rate limiting protection.
    
    Returns:
        webdriver.Firefox: Configured Firefox WebDriver instance
    """
    try:
        # Firefox options optimized for stealth and speed
        options = Options()
        options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-gpu')
        options.add_argument('--disable-extensions')
        options.add_argument('--disable-plugins')
        options.add_argument('--window-size=1920,1080')
        
        # Anti-detection preferences
        options.set_preference('dom.webdriver.enabled', False)
        options.set_preference('useAutomationExtension', False)
        options.set_preference('general.platform.override', 'Win32')
        options.set_preference('general.oscpu.override', 'Windows NT 10.0; Win64; x64')
        
        # Performance and stealth preferences
        options.set_preference('network.http.pipelining', True)
        options.set_preference('network.http.proxy.pipelining', True)
        options.set_preference('network.http.pipelining.maxrequests', 8)
        options.set_preference('content.notify.interval', 500000)
        options.set_preference('content.notify.ontimer', True)
        options.set_preference('content.switch.threshold', 250000)
        options.set_preference('browser.cache.memory.capacity', 65536)
        options.set_preference('browser.startup.homepage', "about:blank")
        options.set_preference('reader.parse-on-load.enabled', False)
        options.set_preference('browser.pocket.enabled', False)
        options.set_preference('loop.enabled', False)
        options.set_preference('browser.chrome.toolbar_tips', False)
        options.set_preference('browser.safebrowsing.enabled', False)
        options.set_preference('browser.safebrowsing.downloads.enabled', False)
        options.set_preference('browser.safebrowsing.malware.enabled', False)
        options.set_preference('webgl.disabled', True)
        options.set_preference('media.peerconnection.enabled', False)
        
        # Set random user agent
        user_agent = get_random_user_agent()
        options.set_preference("general.useragent.override", user_agent)
        logging.info(f"Using User Agent: {user_agent}")
        
        # Try multiple methods to get geckodriver
        driver_path = None
        
        # Method 1: Try webdriver_manager
        try:
            logging.info("Attempting to use GeckoDriverManager...")
            driver_path = GeckoDriverManager().install()
            logging.info(f"GeckoDriverManager successful: {driver_path}")
        except Exception as e:
            logging.warning(f"GeckoDriverManager failed: {e}")
            
        # Method 2: Try manual download if webdriver_manager failed
        if not driver_path:
            logging.info("Attempting manual geckodriver download...")
            driver_path = download_geckodriver_manually()
            
        # Method 3: Try system PATH
        if not driver_path:
            logging.info("Attempting to use geckodriver from system PATH...")
            import shutil
            system_driver = shutil.which('geckodriver')
            if system_driver:
                driver_path = system_driver
                logging.info(f"Found geckodriver in system PATH: {driver_path}")
        
        # Method 4: Try common installation paths
        if not driver_path:
            common_paths = [
                r"C:\Program Files\Mozilla Firefox\geckodriver.exe",
                r"C:\Program Files (x86)\Mozilla Firefox\geckodriver.exe",
                "/usr/local/bin/geckodriver",
                "/usr/bin/geckodriver",
                os.path.join(os.path.expanduser("~"), ".local", "bin", "geckodriver")
            ]
            
            for path in common_paths:
                if os.path.exists(path):
                    driver_path = path
                    logging.info(f"Found geckodriver at common path: {driver_path}")
                    break
        
        if not driver_path:
            raise Exception("Could not locate or download geckodriver. Please install Firefox and geckodriver manually.")
        
        # Test internet connectivity
        try:
            test_response = requests.get("https://www.google.com", timeout=10)
            logging.info("Internet connectivity confirmed")
        except Exception as e:
            logging.warning(f"Internet connectivity issue detected: {e}")
        
        # Setup service
        service = Service(driver_path)
        
        # Create driver with reduced timeouts
        driver = webdriver.Firefox(service=service, options=options)
        driver.implicitly_wait(5)
        driver.set_page_load_timeout(30)
        
        # Execute script to remove webdriver property
        driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        
        logging.info("WebDriver initialized successfully with anti-detection measures")
        return driver
        
    except Exception as e:
        logging.error(f"Failed to initialize WebDriver: {e}")
        
        # Provide helpful error message
        if "Could not reach host" in str(e) or "offline" in str(e).lower():
            logging.error("Network connectivity issue detected. Please check:")
            logging.error("1. Internet connection is working")
            logging.error("2. Firewall/antivirus is not blocking the connection")
            logging.error("3. Proxy settings if applicable")
        
        return None


def get_hotel_links(connection):
    """
    Fetch hotel links from the bronze.hotel_links table.
    
    Args:
        connection: Database connection object
        
    Returns:
        list: List of hotel URLs and names
    """
    try:
        cursor = connection.cursor()
        cursor.execute("SELECT \"Link\", \"Name\" FROM bronze.hotel_links WHERE is_scraped = FALSE OR is_scraped IS NULL")
        hotels = cursor.fetchall()
        cursor.close()
        
        logging.info(f"Retrieved {len(hotels)} hotel links from database")
        return hotels
        
    except Exception as e:
        logging.error(f"Failed to fetch hotel links: {e}")
        return []


def create_reviews_table(connection):
    """
    Create the bronze.hotels_reviews_test table if it doesn't exist.
    
    Args:
        connection: Database connection object
    """
    try:
        cursor = connection.cursor()
        
        create_table_query = """
        CREATE TABLE IF NOT EXISTS bronze.hotels_reviews_test (
            id SERIAL PRIMARY KEY,
            city VARCHAR(255),
            hotel_name VARCHAR(500),
            reviewer_name VARCHAR(255),
            reviewer_nationality VARCHAR(100),
            duration VARCHAR(100),
            check_in_date VARCHAR(100),
            travel_type VARCHAR(100),
            room_type VARCHAR(500),
            review_date VARCHAR(100),
            positive_review TEXT,
            negative_review TEXT,
            scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        
        cursor.execute(create_table_query)
        connection.commit()
        cursor.close()
        
        logging.info("Reviews table created or already exists")
        
    except Exception as e:
        logging.error(f"Error creating reviews table: {e}")
        connection.rollback()


def insert_reviews_to_db(connection, reviews_data):
    """
    Insert review data into the bronze.hotels_reviews_test table with optimized batch processing.
    
    Args:
        connection: Database connection object
        reviews_data: List of review dictionaries
    """
    if not reviews_data:
        return
    
    try:
        cursor = connection.cursor()
        
        # Use psycopg2's execute_values for faster batch inserts
        from psycopg2.extras import execute_values
        
        insert_query = """
        INSERT INTO bronze.hotels_reviews_test 
        (city, hotel_name, reviewer_name, reviewer_nationality, duration, 
         check_in_date, travel_type, room_type, review_date, positive_review, negative_review)
        VALUES %s
        """
        
        # Prepare data for insertion
        insert_data = [
            (
                review['City'],
                review['Hotel Name'],
                review['Reviewer Name'],
                review['Reviewer Nationality'],
                review['Duration'],
                review['Check-in Date'],
                review['Travel Type'],
                review['Room Type'],
                review['Review Date'],
                review['Positive Review'],
                review['Negative Review']
            )
            for review in reviews_data
        ]
        
        # Execute optimized batch insert
        execute_values(cursor, insert_query, insert_data, template=None, page_size=100)
        connection.commit()
        cursor.close()
        
        logging.info(f"Successfully inserted {len(reviews_data)} reviews into database")
        
    except Exception as e:
        logging.error(f"Error inserting reviews to database: {e}")
        connection.rollback()


def smart_delay(base_delay=3, variance=2, request_count=0):
    """
    Implement intelligent delays to avoid rate limiting.
    
    Args:
        base_delay: Base delay in seconds
        variance: Random variance to add
        request_count: Number of requests made (for progressive delays)
    """
    # Progressive delay based on request count
    progressive_factor = min(request_count / 50, 2)  # Max 2x delay after 50 requests
    
    # Calculate delay with exponential backoff for high request counts
    if request_count > 100:
        delay = base_delay * (1.5 ** (request_count / 100)) + random.uniform(0, variance)
    else:
        delay = base_delay + progressive_factor + random.uniform(0, variance)
    
    # Cap maximum delay at 15 seconds
    delay = min(delay, 15)
    
    logging.debug(f"Sleeping for {delay:.2f} seconds (request #{request_count})")
    sleep(delay)


def get_current_quarter():
    """
    Get the previous quarter based on current date.
    
    Returns:
        str: Quarter string that matches Booking.com's format
    """
    current_month = datetime.datetime.now().month
    
    # Determine current quarter and return previous quarter
    if 1 <= current_month <= 3:  # Q1 (Jan-Mar) -> return Q4 of previous year
        return "Dec-Feb"
    elif 4 <= current_month <= 6:  # Q2 (Apr-Jun) -> return Q1
        return "Mar-May"
    elif 7 <= current_month <= 9:  # Q3 (Jul-Sep) -> return Q2
        return "Jun-Aug"
    else:  # Q4 (Oct-Dec) -> return Q3
        return "Sep-Nov"


def select_time_of_year_filter(driver, target_quarter=None):
    """
    Select the time of year filter to get reviews from a specific quarter.
    
    Args:
        driver: WebDriver instance
        target_quarter: Quarter to select (e.g., "Mar-May"). If None, uses previous quarter.
        
    Returns:
        bool: True if filter was successfully applied, False otherwise
    """
    try:
        if not target_quarter:
            target_quarter = get_current_quarter()
        
        logging.info(f"Attempting to select time of year filter: {target_quarter}")
        
        # Wait for the time of year dropdown to be available
        time_dropdown = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, '//select[@name="timeOfYear"]'))
        )
        
        # Make sure dropdown is clickable
        WebDriverWait(driver, 5).until(
            EC.element_to_be_clickable((By.XPATH, '//select[@name="timeOfYear"]'))
        )
        
        # Get all available options
        from selenium.webdriver.support.ui import Select
        select = Select(time_dropdown)
        
        # Log available options for debugging
        options = [option.text for option in select.options]
        logging.info(f"Available time of year options: {options}")
        
        # Try to select the target quarter
        quarter_selected = False
        
        # Try exact match first
        for option in select.options:
            if target_quarter.lower() in option.text.lower() or option.text.lower() in target_quarter.lower():
                select.select_by_visible_text(option.text)
                quarter_selected = True
                logging.info(f"Selected time of year: {option.text}")
                break
        
        # If exact match failed, try partial matches
        if not quarter_selected:
            quarter_parts = target_quarter.split('-')
            if len(quarter_parts) == 2:
                for option in select.options:
                    option_text = option.text.lower()
                    if any(part.lower() in option_text for part in quarter_parts):
                        select.select_by_visible_text(option.text)
                        quarter_selected = True
                        logging.info(f"Selected time of year (partial match): {option.text}")
                        break
        
        if quarter_selected:
            # Wait a moment for the filter to apply
            sleep(2)
            logging.info(f"Successfully applied time of year filter: {target_quarter}")
            return True
        else:
            logging.warning(f"Could not find matching option for quarter: {target_quarter}")
            return False
            
    except Exception as e:
        logging.warning(f"Could not select time of year filter: {e}")
        return False


def scrape_hotel_reviews(driver, hotel_url, hotel_name, connection, max_pages=None):
    """
    Scrape reviews for a single hotel with enhanced rate limiting protection.
    
    Args:
        driver: WebDriver instance
        hotel_url: URL of the hotel
        hotel_name: Name of the hotel
        connection: Database connection object
        max_pages: Maximum number of pages to scrape (None for all pages)
        
    Returns:
        int: Number of reviews scraped
    """
    total_reviews_count = 0
    request_count = 0
    
    try:
        # Navigate to hotel page with retry mechanism
        max_retries = 3
        for attempt in range(max_retries):
            try:
                driver.get(hotel_url)
                request_count += 1
                smart_delay(2, 2, request_count)
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                logging.warning(f"Attempt {attempt + 1} failed for {hotel_name}, retrying...")
                smart_delay(5, 3, request_count)
        
        # Get city information with retry
        city_ = "Unknown City"
        try:
            city_element = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.XPATH, "//a[@class='bui_breadcrumb__link_masked']"))
            )
            city_ = city_element.text
        except:
            # Try alternative selectors
            try:
                city_element = driver.find_element(By.XPATH, "//nav[@aria-label='Breadcrumb']//a[contains(@class,'breadcrumb')]")
                city_ = city_element.text
            except:
                pass
            
        logging.info(f"Scraping reviews for {hotel_name} in {city_}")
        
        # Click "read all reviews" button with multiple attempts
        try:
            # Try multiple possible selectors
            selectors = [
                "//button[@data-testid='fr-read-all-reviews']",
                "//button[contains(text(), 'read all reviews')]",
                "//a[contains(@href, 'reviews')]",
                "//button[contains(text(), 'Read all')]"
            ]
            
            read_all_reviews_btn = None
            for selector in selectors:
                try:
                    read_all_reviews_btn = WebDriverWait(driver, 8).until(
                        EC.element_to_be_clickable((By.XPATH, selector))
                    )
                    break
                except:
                    continue
            
            if read_all_reviews_btn:
                driver.execute_script("arguments[0].click();", read_all_reviews_btn)
                request_count += 1
                smart_delay(3, 2, request_count)
            else:
                logging.warning(f"Could not find 'read all reviews' button for {hotel_name}")
                return total_reviews_count
                
        except Exception as e:
            logging.warning(f"Could not find 'read all reviews' button for {hotel_name}: {e}")
            return total_reviews_count
        
        # Select time of year filter before finding review cards
        select_time_of_year_filter(driver)
        
        page_count = 1
        consecutive_failures = 0
        
        while True:
            # Check max_pages limit
            if max_pages and page_count > max_pages:
                logging.info(f"Reached maximum page limit ({max_pages}) for {hotel_name}")
                break
            
            # Break if too many consecutive failures
            if consecutive_failures >= 3:
                logging.warning(f"Too many consecutive failures, stopping for {hotel_name}")
                break
                
            logging.info(f"Scraping page {page_count} for {hotel_name}")
            page_reviews = []
            
            # Wait for review cards to load with retry
            try:
                WebDriverWait(driver, 15).until(
                    EC.presence_of_element_located((By.XPATH, "//div[@data-testid='review-card']"))
                )
                consecutive_failures = 0  # Reset failure count
            except:
                logging.warning(f"No review cards found on page {page_count}")
                consecutive_failures += 1
                smart_delay(5, 3, request_count)
                continue
            
            # Find all review cards
            review_cards = driver.find_elements(By.XPATH, "//div[@data-testid='review-card']")
            
            if not review_cards:
                logging.info(f"No more review cards found for {hotel_name}")
                break
            
            # Extract data from each card with optimized element finding
            for card_index, card in enumerate(review_cards):
                try:
                    # Add small delay every few cards to avoid being too aggressive
                    if card_index % 5 == 0 and card_index > 0:
                        sleep(random.uniform(0.5, 1.5))
                    
                    # Get reviewer nationality with multiple attempts
                    reviewer_nationality = "Unknown Nationality"
                    try:
                        nationality_selectors = [
                            './/img[contains(@class, "b8d1620349")]',
                            './/img[contains(@alt, "")]',
                            './/span[contains(@class, "nationality")]'
                        ]
                        for selector in nationality_selectors:
                            try:
                                nationality_element = card.find_element(By.XPATH, selector)
                                reviewer_nationality = nationality_element.get_attribute("alt") or nationality_element.text
                                if reviewer_nationality and reviewer_nationality != "":
                                    break
                            except:
                                continue
                    except:
                        pass
                    
                    # Use faster element finding with single queries
                    def safe_get_text(xpath, default="N/A"):
                        try:
                            elements = card.find_elements(By.XPATH, xpath)
                            return elements[0].text if elements else default
                        except:
                            return default
                    
                    review_data = {
                        'City': city_,
                        'Hotel Name': hotel_name,
                        'Reviewer Name': safe_get_text('.//div[@class="b08850ce41 f546354b44"]', "No Name"),
                        'Reviewer Nationality': reviewer_nationality,
                        'Duration': safe_get_text('.//span[@data-testid="review-num-nights"]', "No Duration"),
                        'Check-in Date': safe_get_text('.//span[@data-testid="review-stay-date"]', "No Check-in Date"),
                        'Travel Type': safe_get_text('.//span[@data-testid="review-traveler-type"]', "No Type"),
                        'Room Type': safe_get_text('.//span[@data-testid="review-room-name"]', "No Room Type"),
                        'Review Date': safe_get_text('.//span[@data-testid="review-date"]', "No Date"),
                        'Positive Review': safe_get_text('.//div[@class="b99b6ef58f d14152e7c3"]', "No Positive Feedback"),
                        'Negative Review': safe_get_text('.//div[@data-testid="review-negative-text"]', "No Negative Feedback")
                    }
                    
                    page_reviews.append(review_data)
                    
                except Exception as e:
                    logging.error(f"Error extracting review data: {e}")
                    continue
            
            # Insert page reviews to database incrementally
            if page_reviews:
                insert_reviews_to_db(connection, page_reviews)
                total_reviews_count += len(page_reviews)
                logging.info(f"Inserted {len(page_reviews)} reviews from page {page_count} for {hotel_name}")
            
            # Try to go to next page with enhanced error handling
            try:
                next_page_selectors = [
                    '//button[@aria-label="Next page"]',
                    '//button[contains(@class, "next")]',
                    '//a[contains(@aria-label, "Next")]'
                ]
                
                next_page_btn = None
                for selector in next_page_selectors:
                    try:
                        next_page_btn = WebDriverWait(driver, 5).until(
                            EC.element_to_be_clickable((By.XPATH, selector))
                        )
                        break
                    except:
                        continue
                
                if next_page_btn and next_page_btn.is_enabled():
                    driver.execute_script("arguments[0].click();", next_page_btn)
                    request_count += 1
                    smart_delay(4, 3, request_count)  # Longer delay between pages
                    page_count += 1
                else:
                    logging.info(f"Reached last page for {hotel_name}")
                    break
            except:
                logging.info(f"No more pages available for {hotel_name}")
                break
        
        logging.info(f"Total extracted and inserted {total_reviews_count} reviews for {hotel_name}")
        return total_reviews_count
        
    except Exception as e:
        logging.error(f"Error scraping reviews for {hotel_name}: {e}")
        return total_reviews_count


def mark_hotel_as_scraped(connection, hotel_url):
    """
    Mark a hotel as scraped in the database.
    
    Args:
        connection: Database connection object
        hotel_url: URL of the hotel that was scraped
    """
    try:
        cursor = connection.cursor()
        cursor.execute("UPDATE bronze.hotel_links SET is_scraped = TRUE WHERE \"Link\" = %s", (hotel_url,))
        connection.commit()
        cursor.close()
        logging.info(f"Marked hotel as scraped: {hotel_url}")
    except Exception as e:
        logging.error(f"Error marking hotel as scraped: {e}")
        connection.rollback()


def get_scraping_statistics(connection):
    """
    Get statistics about today's scraping session.
    
    Args:
        connection: Database connection object
        
    Returns:
        dict: Dictionary containing scraping statistics
    """
    try:
        cursor = connection.cursor()
        
        # Get reviews scraped today
        cursor.execute("""
            SELECT COUNT(*) FROM bronze.hotels_reviews_test 
            WHERE scraped_at::date = CURRENT_DATE
        """)
        reviews_today = cursor.fetchone()[0] or 0
        
        # Get total reviews in database
        cursor.execute("SELECT COUNT(*) FROM bronze.hotels_reviews_test")
        total_reviews = cursor.fetchone()[0] or 0
        
        # Get hotels marked as scraped
        cursor.execute("SELECT COUNT(*) FROM bronze.hotel_links WHERE is_scraped = TRUE")
        scraped_hotels = cursor.fetchone()[0] or 0
        
        # Get total hotels in database
        cursor.execute("SELECT COUNT(*) FROM bronze.hotel_links")
        total_hotels = cursor.fetchone()[0] or 0
        
        # Get unscraped hotels
        cursor.execute("""
            SELECT COUNT(*) FROM bronze.hotel_links 
            WHERE is_scraped = FALSE OR is_scraped IS NULL
        """)
        unscraped_hotels = cursor.fetchone()[0] or 0
        
        cursor.close()
        
        return {
            'reviews_today': reviews_today,
            'total_reviews': total_reviews,
            'scraped_hotels': scraped_hotels,
            'total_hotels': total_hotels,
            'unscraped_hotels': unscraped_hotels,
            'completion_rate': (scraped_hotels / total_hotels * 100) if total_hotels > 0 else 0
        }
        
    except Exception as e:
        logging.error(f"Error getting scraping statistics: {e}")
        return {
            'reviews_today': 0,
            'total_reviews': 0,
            'scraped_hotels': 0,
            'total_hotels': 0,
            'unscraped_hotels': 0,
            'completion_rate': 0
        }


def get_runtime_config(schedule_type="test"):
    """
    Get runtime configuration based on schedule type.
    
    Args:
        schedule_type: Type of schedule ("quarterly", "daily", etc.)
        
    Returns:
        dict: Configuration dictionary with max_hotels and max_pages_per_hotel
    """
    configs = {
        "quarterly": {
            "max_hotels": None,  # Process all hotels
            "max_pages_per_hotel": None  # Process all pages
        },
        "daily": {
            "max_hotels": 10,  # Limited for daily runs
            "max_pages_per_hotel": 5  # Limited pages for daily runs
        },
        "test": {
            "max_hotels": 2,  # Very limited for testing
            "max_pages_per_hotel": 2  # 2 pages for testing
        }
    }
    
    return configs.get(schedule_type, configs["quarterly"])


def main_scraping_process(max_hotels=None, max_pages_per_hotel=None):
    """
    Main function to orchestrate the hotel review scraping process with enhanced rate limiting.
    
    Args:
        max_hotels: Maximum number of hotels to process (None for all)
        max_pages_per_hotel: Maximum pages per hotel (None for all pages)
        
    Returns:
        dict: Dictionary containing scraping results and statistics
    """
    # Initialize logging
    initialize_logging()
    
    # Create database connection
    connection = create_db_connection()
    if not connection:
        logging.error("Cannot proceed without database connection")
        return {'success': False, 'error': 'Database connection failed'}
    
    # Get initial statistics
    initial_stats = get_scraping_statistics(connection)
    logging.info(f"Initial statistics: {initial_stats}")
    
    # Create reviews table
    create_reviews_table(connection)
    
    # Get hotel links
    hotels = get_hotel_links(connection)
    if not hotels:
        logging.error("No hotel links found in database")
        connection.close()
        return {'success': False, 'error': 'No hotel links found'}
    
    # Shuffle hotels to avoid patterns
    random.shuffle(hotels)
    
    # Limit hotels if specified
    original_hotel_count = len(hotels)
    if max_hotels:
        hotels = hotels[:max_hotels]
        logging.info(f"Limited to processing {max_hotels} hotels out of {original_hotel_count} available")
    
    # Setup driver
    driver = setup_driver()
    if not driver:
        logging.error("Cannot proceed without WebDriver")
        connection.close()
        return {'success': False, 'error': 'WebDriver setup failed'}
    
    total_reviews_scraped = 0
    hotels_processed = 0
    hotels_failed = 0
    
    try:
        for hotel_url, hotel_name in hotels:
            hotels_processed += 1
            logging.info(f"Processing hotel {hotels_processed}/{len(hotels)}: {hotel_name}")
            
            # Recreate driver every 10 hotels to avoid detection
            if hotels_processed % 10 == 0:
                logging.info("Recreating driver for fresh session...")
                driver.quit()
                smart_delay(10, 5, hotels_processed)  # Longer break
                driver = setup_driver()
                if not driver:
                    logging.error("Failed to recreate driver, stopping")
                    hotels_failed += 1
                    break
            
            try:
                # Scrape reviews for current hotel with page limit
                hotel_review_count = scrape_hotel_reviews(
                    driver, hotel_url, hotel_name, connection, max_pages_per_hotel
                )
                total_reviews_scraped += hotel_review_count
                
                # Mark hotel as scraped regardless of success/failure
                mark_hotel_as_scraped(connection, hotel_url)
                
                logging.info(f"Completed {hotel_name}. Reviews: {hotel_review_count}. Total so far: {total_reviews_scraped}")
                
                # Enhanced delay between hotels with progressive increase
                base_delay = 8 + (hotels_processed // 5)  # Increase delay every 5 hotels
                smart_delay(base_delay, 5, hotels_processed)
                
            except Exception as e:
                hotels_failed += 1
                logging.error(f"Failed to process hotel {hotel_name}: {e}")
                
                # Mark as scraped even if failed to avoid reprocessing
                try:
                    mark_hotel_as_scraped(connection, hotel_url)
                except:
                    pass
                
                continue
        
        # Get final statistics
        final_stats = get_scraping_statistics(connection)
        logging.info(f"Final statistics: {final_stats}")
        
        # Calculate session results
        session_results = {
            'success': True,
            'hotels_processed': hotels_processed,
            'hotels_failed': hotels_failed,
            'hotels_success': hotels_processed - hotels_failed,
            'reviews_scraped_session': total_reviews_scraped,
            'reviews_scraped_today': final_stats['reviews_today'],
            'initial_stats': initial_stats,
            'final_stats': final_stats,
            'completion_rate': final_stats['completion_rate']
        }
        
        logging.info(f"Scraping session completed successfully: {session_results}")
        return session_results
            
    except Exception as e:
        logging.error(f"Error in main scraping process: {e}")
        return {
            'success': False,
            'error': str(e),
            'hotels_processed': hotels_processed,
            'hotels_failed': hotels_failed,
            'reviews_scraped_session': total_reviews_scraped
        }
    
    finally:
        # Cleanup
        if driver:
            driver.quit()
        if connection:
            connection.close()
        logging.info("Scraping process cleanup completed")


if __name__ == "__main__":
    # Optional: Limit processing for faster testing
    main_scraping_process(max_hotels=5, max_pages_per_hotel=2)
    #main_scraping_process()


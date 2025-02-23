from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta

import os
import csv
import random
import logging
from time import sleep
import pandas as pd
import requests
from bs4 import BeautifulSoup as bs
from requests.exceptions import RequestException

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.firefox import GeckoDriverManager

import user_agent as ua
from user_agent import generate_user_agent

Base = "C:\\Users\\Kariem\\Desktop\\New_folder\\Smart-Tourism-Dev-Sys\\Data_collecting\\Hotels"
# -----------------------------
# Logging Setup
# -----------------------------
logging.basicConfig(filename=r'C:\Users\Kariem\Desktop\New_folder\Smart-Tourism-Dev-Sys\Data_collecting\Hotels\scraper.log', level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

# -----------------------------
# Global Variables and File Paths
# -----------------------------
total_requests = 0

# User-Agent Headers

HEADERS_LIST = [
    {'User-Agent': generate_user_agent()} for _ in range(10)
]

Links_csv = f'{Base}\\hotels_links_final.csv'
HOTEL_DATA_FILE = f'{Base}\\hotels_data_final.csv'
REVIEWS_FILE = f'{Base}\\hotels_reviews_final.csv'


# -----------------------------
# Utility: Safe HTTP Request
# -----------------------------
def safe_request(url, headers):
    global total_requests
    MAX_RETRIES = 3
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(url, headers=headers, timeout=10)
            total_requests += 1
            response.raise_for_status()
            return response
        except RequestException as e:
            logging.warning(f"Attempt {attempt + 1} failed for {url}: {e}")
            sleep(random.uniform(2, 5))  # Random delay to avoid detection
    return None
 
# -----------------------------
# Function 1: Scrape Hotel Links per City (Handles Pagination)
# -----------------------------
def get_hotels_in_each_city(cities, Cities_url):
    logging.info("Scraping hotel links in each city started")
    hotel_links = []
    global total_requests

    for base_url, city in zip(Cities_url, cities):
        logging.info(f"Scraping hotels for city: {city}")
        page_number = 0

        for _ in range(5):  # Limit to 5 pages per city (each page has 25 hotels)
            url = f"{base_url}&offset={page_number * 25}"  # Adjust offset for pagination
            headers = random.choice(HEADERS_LIST)
            response = safe_request(url, headers)
            
            if not response:
                logging.error(f"Failed to retrieve data for {city}")
                break

            soup = bs(response.content, 'html.parser')
            city_hotel_links = []

            for container in soup.find_all('div', {'data-testid': 'title'}):
                try:
                    hotel_name = container.get_text(strip=True)
                    parent = container.find_parent('div')
                    hotel_link_tag = parent.find('a', href=True)
                    
                    if hotel_link_tag:
                        link = hotel_link_tag['href']
                        if not link.startswith("http"):
                            link = "https://www.booking.com" + link
                        city_hotel_links.append({
                            'Name': hotel_name,
                            'Link': link,
                            'Is_Scraped': False
                        })
                except Exception as e:
                    logging.error(f"Error extracting hotel details in {city}: {e}")

            if not city_hotel_links:
                logging.info(f"No more hotels found on page {page_number + 1} for {city}")
                break  # No more hotels, break the pagination loop
            else:
                logging.info(f"Scraped {len(city_hotel_links)} hotels from page {page_number + 1} in {city}")
                hotel_links.extend(city_hotel_links)

            page_number += 1
            sleep(random.uniform(2, 5))  # Randomized delay to avoid IP bans

    # Write the hotel links to CSV immediately
    pd.DataFrame(hotel_links).to_csv(Links_csv, index=False)
    logging.info(f"Total HTTP requests made: {total_requests}")
    return hotel_links

# -----------------------------
# CSV Initialization for Detailed Data & Reviews
# -----------------------------
def init_csv_files():
    hotel_fieldnames = ['City', 'Hotel_Name', 'Hotel_over_all_rate', 'Staff', 'Facilities', 
                        'Cleanliness', 'Comfort', 'Value_for_money', 'Location', 'Free_Wifi']
    review_fieldnames = ['City', 'Hotel Name', 'Reviewer Name', 'Reviewer Nationality', 
                         'Duration', 'Check-in Date', 'Travel Type', 'Room Type', 
                         'Review Date', 'Positive Review', 'Negative Review']

    # Initialize hotel data file
    if not os.path.exists(HOTEL_DATA_FILE):
        with open(HOTEL_DATA_FILE, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=hotel_fieldnames)
            writer.writeheader()
    
    # Initialize reviews data file
    if not os.path.exists(REVIEWS_FILE):
        with open(REVIEWS_FILE, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=review_fieldnames)
            writer.writeheader()

    return hotel_fieldnames, review_fieldnames

# -----------------------------
# Function 2: Process Hotels Incrementally & Update Links CSV
# -----------------------------
def process_hotels_incrementally_from_file():
    df = pd.read_csv(Links_csv)
    hotel_links = df.to_dict('records')
    hotel_fieldnames, review_fieldnames = init_csv_files()

    # Set up headless Firefox
    options = Options()
    options.add_argument('--headless')
    
    for idx, hotel in enumerate(hotel_links):
        if not str(hotel.get('Is_Scraped', False)).lower() in ["true", "1"]:
            hotel_name = hotel['Name']
            hotel_link = hotel['Link']
            logging.info(f"Processing hotel: {hotel_name}")

            try:
                driver = webdriver.Firefox(service=Service(GeckoDriverManager().install()), options=options)
                driver.get(hotel_link)
                sleep(random.uniform(3, 5))  # Wait for page load with a randomized delay

                # Extract city name
                try:
                    city_ = driver.find_element(By.XPATH, '//a[@class="bui_breadcrumb__link_masked"]').text
                except Exception:
                    city_ = "No City"

                # Extract overall hotel rating
                try:
                    overall_element = driver.find_element(By.CSS_SELECTOR, 'div[data-testid="review-score-right-component"] .ac4a7896c7')
                    hotel_over_all_rate = overall_element.text.strip()
                except Exception:
                    hotel_over_all_rate = "No Rate"

                # Extract category ratings
                cat_dict = {}
                try:
                    category_elements = driver.find_elements(By.XPATH, '//div[@data-testid="review-subscore"]')
                    for cat in category_elements:
                        label = cat.find_element(By.CLASS_NAME, 'ccb65902b2').text.strip().lower()
                        rating = cat.find_element(By.CLASS_NAME, 'bdc1ea4a28').text.strip()
                        cat_dict[label] = rating
                except Exception as e:
                    logging.warning(f"Error extracting categories for {hotel_name}: {e}")

                # Compile hotel data
                hotel_data = {
                    'City': city_,
                    'Hotel_Name': hotel_name,
                    'Hotel_over_all_rate': hotel_over_all_rate,
                    'Staff': cat_dict.get("staff", "No Rate"),
                    'Facilities': cat_dict.get("facilities", "No Rate"),
                    'Cleanliness': cat_dict.get("cleanliness", "No Rate"),
                    'Comfort': cat_dict.get("comfort", "No Rate"),
                    'Value_for_money': cat_dict.get("value for money", "No Rate"),
                    'Location': cat_dict.get("location", "No Rate"),
                    'Free_Wifi': cat_dict.get("free wifi", "No Rate")
                }

                # Save hotel data
                with open(HOTEL_DATA_FILE, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=hotel_fieldnames)
                    writer.writerow(hotel_data)

                # Process reviews
                process_reviews(driver, hotel_name, city_, review_fieldnames)
                            # Mark hotel as scraped
                hotel_links[idx]['Is_Scraped'] = True
                pd.DataFrame(hotel_links).to_csv(Links_csv, index=False)
                logging.info(f"Marked {hotel_name} as scraped.")

            except Exception as e:
                logging.error(f"Error processing hotel {hotel_name}: {e}")
                if "API rate limit exceeded" in str(e):
                    logging.error("API rate limit exceeded. Exiting.")
                    driver.quit()
                    break
            finally:
                driver.quit()


            

    logging.info("Incremental processing complete.")

# -----------------------------
# Function 3: Process Reviews (Handles Pagination)
# -----------------------------
def process_reviews(driver, hotel_name, city_, review_fieldnames):
    try:
        reviews_button = driver.find_element(By.XPATH, '//button[@data-testid="fr-read-all-reviews"]')
        reviews_button.click()
        sleep(random.uniform(3, 5))
    except Exception:
        logging.info(f"No 'Read All Reviews' button for {hotel_name}")
    
    reviews_num = 0

    for _ in range(10):        
        reviews_cards = driver.find_elements(By.XPATH, '//div[@data-testid="review-card"]')
        reviews_num += 10
        for card in reviews_cards:
            try:
                review_data = {
                    'City': city_,
                    "Hotel Name": hotel_name,
                    'Reviewer Name': card.find_element(By.CLASS_NAME, 'e6208ee469').text if card.find_elements(By.CLASS_NAME, 'e6208ee469') else "No Name",
                    'Reviewer Nationality': card.find_element(By.CLASS_NAME, 'f45d8e4c32').text if card.find_elements(By.CLASS_NAME, 'f45d8e4c32') else "No Nationality",
                    'Duration': card.find_element(By.XPATH, './/span[@data-testid="review-num-nights"]').text if card.find_elements(By.XPATH, './/span[@data-testid="review-num-nights"]') else "No Duration",
                    'Check-in Date': card.find_element(By.XPATH, './/span[@data-testid="review-stay-date"]').text if card.find_elements(By.XPATH, './/span[@data-testid="review-stay-date"]') else "No Check-in Date",
                    'Travel Type': card.find_element(By.XPATH, './/span[@data-testid="review-traveler-type"]').text if card.find_elements(By.XPATH, './/span[@data-testid="review-traveler-type"]') else "No Type",
                    'Room Type': card.find_element(By.XPATH, './/span[@data-testid="review-room-name"]').text if card.find_elements(By.XPATH, './/span[@data-testid="review-room-name"]') else "No Room Type",
                    'Review Date': card.find_element(By.XPATH, './/span[@data-testid="review-date"]').text if card.find_elements(By.XPATH, './/span[@data-testid="review-date"]') else "No Date",
                    'Positive Review': card.find_element(By.CLASS_NAME, 'b5726afd0b').text if card.find_elements(By.CLASS_NAME, 'b5726afd0b') else "No Positive Feedback",
                    'Negative Review': card.find_element(By.XPATH, './/div[@data-testid="review-negative-text"]').text if card.find_elements(By.XPATH, './/div[@data-testid="review-negative-text"]') else "No Negative Feedback"
                }
                with open(REVIEWS_FILE, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=review_fieldnames)
                    writer.writerow(review_data)
                    
            except Exception as e:
                logging.warning(f"Error extracting review for {hotel_name}: {e}")


        # Handle pagination
        try:
            next_page_btn = driver.find_element(By.XPATH,'//button[@aria-label="Next page"]')
            next_page_btn.click()
            sleep(random.uniform(3, 5))
        except Exception:
            logging.info(f"No more review pages for {hotel_name}")
            break
    logging.info(f"Processed {reviews_num}reviews for {hotel_name}")
# -----------------------------
# Main Execution
# -----------------------------
if __name__ == "__main__":
    cities = ['Luxor','Hurghada','Sharm_ElSheikh', 'Aswan']
    Cities_url = [
        'https://www.booking.com/searchresults.html?ss=Luxor%2C+Egypt&efdco=1&label=gen173nr-1FCAEoggI46AdIM1gEaEOIAQGYATG4ARfIAQ_YAQHoAQH4AQKIAgGoAgO4AoaW3L0GwAIB0gIkYmU1OTg0NDQtMTI2Zi00ZGViLWIzZTctZmU3Mjk4YjQ4MGI12AIF4AIB&aid=304142&lang=en-us&sb=1&src_elem=sb&src=index&dest_id=-290821&dest_type=city&ac_position=0&ac_click_type=b&ac_langcode=en&ac_suggestion_list_length=5&search_selected=true&search_pageview_id=427d4d43569d01d9&ac_meta=GhA0MjdkNGQ0MzU2OWQwMWQ5IAAoATICZW46BWx1eG9yQABKAFAA&group_adults=2&no_rooms=1&group_children=0',
        'https://www.booking.com/searchresults.html?ss=Hurghada%2C+Egypt&ssne=Luxor&ssne_untouched=Luxor&efdco=1&label=gen173nr-1FCAEoggI46AdIM1gEaEOIAQGYATG4ARfIAQ_YAQHoAQH4AQKIAgGoAgO4AoaW3L0GwAIB0gIkYmU1OTg0NDQtMTI2Zi00ZGViLWIzZTctZmU3Mjk4YjQ4MGI12AIF4AIB&aid=304142&lang=en-us&sb=1&src_elem=sb&src=searchresults&dest_id=-290029&dest_type=city&group_adults=2&no_rooms=1&group_children=0',
        'https://www.booking.com/searchresults.html?ss=shar&label=gen173nr-1FCAEoggI46AdIM1gEaEOIAQGYATG4ARfIAQ_YAQHoAQH4AQKIAgGoAgO4AoDku70GwAIB0gIkZmE4MTAxNmItZGRiOC00ZGNjLWIxZDUtM2RjZGQ0ZWYxNmRl2AIF4AIB&aid=304142&lang=en-us&sb=1&src_elem=sb&src=index&dest_id=-302053&dest_type=city&ac_position=0&ac_click_type=b&ac_langcode=en&ac_suggestion_list_length=5&search_selected=true&search_pageview_id=68cd35409ba1044c&ac_meta=GhA2OGNkMzU0MDliYTEwNDRjIAAoATICZW46BHNoYXJAAEoAUAA%3D&group_adults=2&no_rooms=1&group_children=0',
        'https://www.booking.com/searchresults.html?ss=Aswan%2C+Aswan+Governorate%2C+Egypt&ssne=Sharm+El+Sheikh&ssne_untouched=Sharm+El+Sheikh&efdco=1&label=gen173nr-1FCAEoggI46AdIM1gEaEOIAQGYATG4ARfIAQ_YAQHoAQH4AQKIAgGoAgO4AoDku70GwAIB0gIkZmE4MTAxNmItZGRiOC00ZGNjLWIxZDUtM2RjZGQ0ZWYxNmRl2AIF4AIB&aid=304142&lang=en-us&sb=1&src_elem=sb&src=searchresults&dest_id=-291535&dest_type=city&ac_position=0&ac_click_type=b&ac_langcode=en&ac_suggestion_list_length=5&search_selected=true&search_pageview_id=685235446365131f&ac_meta=GhA2ODUyMzU0NDYzNjUxMzFmIAAoATICZW46A2Fzd0AASgBQAA%3D%3D&group_adults=2&no_rooms=1&group_children=0'
    ]
    
    # Step 1: Scrape hotel links per city
    #initialy step to create dataset
    #get_hotels_in_each_city(cities, Cities_url)
    
    # Step 2: Process hotels incrementally from the hotels_links file
    process_hotels_incrementally_from_file()

    logging.info(f"Scraping complete. Total HTTP requests made: {total_requests}")


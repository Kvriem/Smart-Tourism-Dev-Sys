import os
import csv
import random
from time import sleep
import pandas as pd
import requests
from bs4 import BeautifulSoup as bs
from requests.exceptions import RequestException

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.service import Service
from webdriver_manager.firefox import GeckoDriverManager

# -----------------------------
# Global variables and file paths
# -----------------------------
total_requests = 0
HEADERS_LIST = [
    {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"},
    {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0"},
    {"User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:88.0) Gecko/20100101 Firefox/88.0"}
]

LINKS_CSV = r'C:\Users\Kariem\Desktop\Capstone_Porject\repo\Smart-Tourism-Dev-Sys\Data_collecting\Hotels\new_test\hotels_links.csv'
HOTEL_DATA_FILE = r'C:\Users\Kariem\Desktop\Capstone_Porject\repo\Smart-Tourism-Dev-Sys\Data_collecting\Hotels\new_test\hotels_data.csv'
REVIEWS_FILE = r'C:\Users\Kariem\Desktop\Capstone_Porject\repo\Smart-Tourism-Dev-Sys\Data_collecting\Hotels\new_test\hotels_reviews3.csv'

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
            print(f"Attempt {attempt+1} failed for {url}: {e}")
            sleep(2)
    return None

# -----------------------------
# Function 1: Scrape Hotel Links per City
# -----------------------------
def get_hotels_in_each_city(cities, Cities_url):
    print("Scraping hotel links in each city started")
    hotel_links = []
    global total_requests
    for url, city in zip(Cities_url, cities):
        try:
            headers = random.choice(HEADERS_LIST)
            response = safe_request(url, headers)
            if not response:
                print(f"Failed to retrieve data for {city}")
                continue
            print(city, "Connection success:", response.status_code)
            soup = bs(response.content, 'html.parser')
            
            city_hotel_links = []
            for container in soup.find_all('div', {'data-testid': 'title'}):
                try:
                    hotel_name = container.get_text(strip=True)
                    parent = container.find_parent('div')
                    hotel_link_tag = parent.find('a', href=True)
                    
                    # Extract review rating if available
                    rating_container = parent.find('div', {'data-testid': 'review-score'})
                    hotel_rate = rating_container.get_text(strip=True) if rating_container else 'N/A'
                    
                    if hotel_link_tag:
                        link = hotel_link_tag['href']
                        if not link.startswith("http"):
                            link = "https://www.booking.com" + link
                        city_hotel_links.append({
                            'Name': hotel_name,
                            'Link': link,
                            'Rate': hotel_rate,
                            'Is_Scraped': False
                        })
                except Exception as e:
                    print(f"Error extracting hotel details in {city}: {e}")
            
            if not city_hotel_links:
                print(f"No hotels found in {city}")
            else:
                print("Hotels links and rates scraped successfully for", city)
            hotel_links.extend(city_hotel_links)
        
        except Exception as e:
            print(f"Connection error for {city}: {e}")
    
    # Write the hotel links to CSV immediately
    df_links = pd.DataFrame(hotel_links)
    df_links.to_csv(LINKS_CSV, index=False)
    print(f"Total HTTP requests made: {total_requests}")
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
    with open(HOTEL_DATA_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=hotel_fieldnames)
        writer.writeheader()
    with open(REVIEWS_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=review_fieldnames)
        writer.writeheader()
    return hotel_fieldnames, review_fieldnames

# -----------------------------
# Function 2: Process Hotels Incrementally & Update Links CSV
# -----------------------------
def process_hotels_incrementally_from_file():
    # Read the hotels_links CSV file into a dataframe and then into a list of dicts.
    df = pd.read_csv(LINKS_CSV)
    hotel_links = df.to_dict('records')
    
    hotel_fieldnames, review_fieldnames = init_csv_files()  # reinitialize detailed CSVs
    
    for idx, hotel in enumerate(hotel_links):
        if not str(hotel.get('Is_Scraped', False)).lower() in ["true", "1"]:
            hotelName = hotel['Name']
            hotelLink = hotel['Link']
            print(f"Processing hotel: {hotelName}")
            try:
                driver = webdriver.Firefox(service=Service(GeckoDriverManager().install()))
                driver.get(hotelLink)
                sleep(3)  # Wait for page load

                # Extract city name
                try:
                    city_ = driver.find_element(By.XPATH, '//a[@class="bui_breadcrumb__link_masked"]').text
                except Exception:
                    city_ = "No City"

                # Extract overall hotel rating using the new selector
                try:
                    overall_element = driver.find_element(By.CSS_SELECTOR, 'div[data-testid="review-score-right-component"] .ac4a7896c7')
                    overall_text = overall_element.text.strip()  # e.g., "Scored 8.5"
                    hotel_over_all_rate = ''.join(ch for ch in overall_text if ch.isdigit() or ch == '.')
                    if not hotel_over_all_rate:
                        hotel_over_all_rate = "No Rate"
                except Exception:
                    hotel_over_all_rate = "No Rate"

                # --- Extract category ratings ---
                try:
                    cat_dict = {}
                    # Find all category containers
                    category_elements = driver.find_elements(
                        By.XPATH,
                        '//div[contains(@class, "b817090550") and contains(@class, "a7cf1a6b1d") and .//div[@data-testid="review-subscore"]]'
                    )
                    for cat in category_elements:
                        try:
                            # Get label element (the one without "bdc1ea4a28")
                            label_el = cat.find_element(
                                By.XPATH,
                                './/div[contains(@class, "ccb65902b2") and not(contains(@class, "bdc1ea4a28"))]'
                            )
                            label_text = label_el.text.strip().lower()
                            # Get the rating value (the element with "bdc1ea4a28")
                            rating_el = cat.find_element(
                                By.XPATH,
                                './/div[contains(@class, "bdc1ea4a28")]'
                            )
                            rating_text = rating_el.text.strip()
                            cat_dict[label_text] = rating_text
                        except Exception as inner_e:
                            print(f"Error extracting one category for {hotelName}: {inner_e}")
                    Staff = cat_dict.get("staff", "No Rate")
                    Facilities = cat_dict.get("facilities", "No Rate")
                    Cleanliness = cat_dict.get("cleanliness", "No Rate")
                    Comfort = cat_dict.get("comfort", "No Rate")
                    Value_for_money = cat_dict.get("value for money", "No Rate")
                    Location = cat_dict.get("location", "No Rate")
                    Free_Wifi = cat_dict.get("free wifi", "No Rate")
                except Exception as e:
                    print(f"Error extracting categories for {hotelName}: {e}")
                    Staff = Facilities = Cleanliness = Comfort = Value_for_money = Location = Free_Wifi = "No Rate"

                # Build hotel detailed data
                hotel_data = {
                    'City': city_,
                    'Hotel_Name': hotelName,
                    'Hotel_over_all_rate': hotel_over_all_rate,
                    'Staff': Staff,
                    'Facilities': Facilities,
                    'Cleanliness': Cleanliness,
                    'Comfort': Comfort,
                    'Value_for_money': Value_for_money,
                    'Location': Location,
                    'Free_Wifi': Free_Wifi
                }
                # Append hotel data immediately
                with open(HOTEL_DATA_FILE, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=hotel_fieldnames)
                    writer.writerow(hotel_data)

                # Process reviews
                try:
                    reviews_button = driver.find_element(By.XPATH, '//button[@data-testid="fr-read-all-reviews"]')
                    reviews_button.click()
                    sleep(3)
                except Exception as e:
                    print(f"Review button issue for {hotelName}: {e}")

                reviews_cards = driver.find_elements(By.XPATH, '//div[@data-testid="review-card"]')
                for card in reviews_cards:
                    review_data = {
                        'City': city_,
                        "Hotel Name": hotelName,
                        'Reviewer Name': (card.find_element(By.XPATH, './/div[@aria-label="Reviewer"]//div[contains(@class,"e6208ee469")]').text 
                                          if card.find_elements(By.XPATH, './/div[@aria-label="Reviewer"]//div[contains(@class,"e6208ee469")]') 
                                          else "No Name"),
                        'Reviewer Nationality': (card.find_element(By.XPATH, './/div[contains(@class,"f45d8e4c32")]//span').text 
                                                 if card.find_elements(By.XPATH, './/div[contains(@class,"f45d8e4c32")]//span') 
                                                 else "No Nationality"),
                        'Duration': (card.find_element(By.XPATH, './/span[@data-testid="review-num-nights"]').text 
                                     if card.find_elements(By.XPATH, './/span[@data-testid="review-num-nights"]') 
                                     else "No Duration"),
                        'Check-in Date': (card.find_element(By.XPATH, './/span[@data-testid="review-stay-date"]').text 
                                          if card.find_elements(By.XPATH, './/span[@data-testid="review-stay-date"]') 
                                          else "No Check-in Date"),
                        'Travel Type': (card.find_element(By.XPATH, './/span[@data-testid="review-traveler-type"]').text 
                                        if card.find_elements(By.XPATH, './/span[@data-testid="review-traveler-type"]') 
                                        else "No Type"),
                        'Room Type': (card.find_element(By.XPATH, './/span[@data-testid="review-room-name"]').text 
                                      if card.find_elements(By.XPATH, './/span[@data-testid="review-room-name"]') 
                                      else "No Room Type"),
                        'Review Date': (card.find_element(By.XPATH, './/span[@data-testid="review-date"]').text 
                                        if card.find_elements(By.XPATH, './/span[@data-testid="review-date"]') 
                                        else "No Date"),
                        'Positive Review': (card.find_element(By.XPATH, './/div[@aria-label="Review"]//div[contains(@class,"b5726afd0b")]').text 
                                            if card.find_elements(By.XPATH, './/div[@aria-label="Review"]//div[contains(@class,"b5726afd0b")]') 
                                            else "No Positive Feedback"),
                        'Negative Review': (card.find_element(By.XPATH, './/div[@aria-label="Review"]//div[@data-testid="review-negative-text"]//div[contains(@class,"b5726afd0b")]').text 
                                            if card.find_elements(By.XPATH, './/div[@aria-label="Review"]//div[@data-testid="review-negative-text"]//div[contains(@class,"b5726afd0b")]') 
                                            else "No Negative Feedback")
                    }
                    with open(REVIEWS_FILE, 'a', newline='', encoding='utf-8') as f:
                        writer = csv.DictWriter(f, fieldnames=review_fieldnames)
                        writer.writerow(review_data)
            except Exception as e:
                print(f"Error processing hotel {hotelName}: {e}")
            finally:
                driver.quit()
            
            # Update the current hotel's flag in the list and write back to CSV.
            hotel_links[idx]['Is_Scraped'] = True
            df_updated = pd.DataFrame(hotel_links)
            df_updated.to_csv(LINKS_CSV, index=False)
            print(f"Updated {hotelName} as scraped in hotels_links.csv")
    print("Incremental processing complete.")

# -----------------------------
# Main Execution
# -----------------------------
if __name__ == "__main__":
    # Example cities and URLs (adjust query parameters accordingly)
    cities = ['Luxor', 'Hurghada']
    Cities_url = [
        'https://www.booking.com/searchresults.html?ss=Luxor%2C+Egypt&...your_query_params...',
        'https://www.booking.com/searchresults.html?ss=Hurghada%2C+Egypt&...your_query_params...'
    ]
    
    # Step 1: (Optional) Scrape hotel links per city if needed.
    # Uncomment the following line if you want to refresh the links file:
    # get_hotels_in_each_city(cities, Cities_url)
    
    # Step 2: Process hotels incrementally from the hotels_links file.
    process_hotels_incrementally_from_file()

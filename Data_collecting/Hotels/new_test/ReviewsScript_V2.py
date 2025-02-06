import os
import csv
from time import sleep
import pandas as pd
from selenium import webdriver
from selenium.webdriver.firefox.service import Service
from webdriver_manager.firefox import GeckoDriverManager 
from selenium.webdriver.common.by import By

# File paths for incremental output
HOTEL_DATA_FILE = r'C:\Users\Kariem\Desktop\Capstone_Porject\repo\Smart-Tourism-Dev-Sys\Data_collecting\Hotels\new_test\hotels_data_test.csv'
REVIEWS_FILE = r'C:\Users\Kariem\Desktop\Capstone_Porject\repo\Smart-Tourism-Dev-Sys\Data_collecting\Hotels\new_test\hotels_reviews_test.csv'

# Write headers for the CSV files only once.
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

# Process each hotel individually and immediately append its data.
def process_hotels_incrementally(hotel_links):
    # Initialize CSV files (this writes the header once)
    hotel_fieldnames, review_fieldnames = init_csv_files()
    
    for hotel in hotel_links:
        if not hotel.get('Is_Scraped', False):
            hotelName = hotel['Name']
            hotelLink = hotel['Link']
            hotel['Is_Scraped'] = True

            try:
                # Launch a new Firefox instance for this hotel
                driver = webdriver.Firefox(service=Service(GeckoDriverManager().install()))
                driver.get(hotelLink)
                sleep(3)  # wait for the page to load

                # Extract city name
                try:
                    city_ = driver.find_element(By.XPATH, '//a[@class="bui_breadcrumb__link_masked"]').text
                except Exception:
                    city_ = "No City"

                # Extract overall hotel rating
                try:
                    hotel_over_all_rate = driver.find_element(By.XPATH, '//div[@class="ac4a7896c7"]').text[0]
                except Exception:
                    hotel_over_all_rate = "No Rate"

                # Extract category ratings using a relative XPath
                try:
                    categories = driver.find_elements(By.XPATH, '//div[@class="b817090550 a7cf1a6b1d"]')
                    if categories:
                        rate_text = categories[0].find_element(By.XPATH, './/div[@class="ccb65902b2 bdc1ea4a28"]').text.strip()
                        # Assume the rate_text is a whitespace-separated string like: "4.5 4.2 4.8 4.3 4.0 4.1 4.2"
                        rates = rate_text.split()
                        if len(rates) >= 7:
                            Staff = rates[0]
                            Facilities = rates[1]
                            Cleanliness = rates[2]
                            Comfort = rates[3]
                            Value_for_money = rates[4]
                            Location = rates[5]
                            Free_Wifi = rates[6]
                        else:
                            Staff = Facilities = Cleanliness = Comfort = Value_for_money = Location = Free_Wifi = "No Rate"
                    else:
                        Staff = Facilities = Cleanliness = Comfort = Value_for_money = Location = Free_Wifi = "No Rate"
                except Exception as e:
                    print(f"Error extracting category ratings for {hotelName}: {e}")
                    Staff = Facilities = Cleanliness = Comfort = Value_for_money = Location = Free_Wifi = "No Rate"

                # Build hotel data dictionary
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
                # Immediately write hotel data to CSV
                with open(HOTEL_DATA_FILE, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=hotel_fieldnames)
                    writer.writerow(hotel_data)

                # Attempt to click "Read All Reviews" if the button is available
                try:
                    reviews_button = driver.find_element(By.XPATH, '//button[@data-testid="fr-read-all-reviews"]')
                    reviews_button.click()
                    sleep(3)
                except Exception as e:
                    print(f"Review button issue for {hotelName}: {e}")

                # Process and write reviews incrementally
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
                    # Immediately write each review to the reviews CSV file
                    with open(REVIEWS_FILE, 'a', newline='', encoding='utf-8') as f:
                        writer = csv.DictWriter(f, fieldnames=review_fieldnames)
                        writer.writerow(review_data)

            except Exception as e:
                print(f"Error processing hotel {hotelName}: {e}")
            finally:
                driver.quit()

    print("Processing complete.")

# Example usage:
if __name__ == "__main__":
    # Load your previously scraped hotel links from CSV
    hotels_df = pd.read_csv(r'C:\Users\Kariem\Desktop\Capstone_Porject\repo\Smart-Tourism-Dev-Sys\Data_collecting\Hotels\hotels_links.csv')
    hotel_links = hotels_df.to_dict('records')
    process_hotels_incrementally(hotel_links)

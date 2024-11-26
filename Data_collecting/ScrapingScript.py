from bs4 import BeautifulSoup as bs
import requests
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.firefox import GeckoDriverManager 
import random

# User-Agent for requests
HEADERS = {
    "User-Agent": random.choice([
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.93 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0",
    ])
}

cities=['Hurghada','Sharm El Sheikh','Ain Sokhna', 'Alexandria', 'Cairo']
Cities_url=['https://www.booking.com/searchresults.html?ss=Hurghada%2C+Egypt&ssne=Alexandria&ssne_untouched=Alexandria&highlighted_hotels=788831&efdco=1&label=gen173nr-1FCAEoggI46AdIM1gEaEOIAQGYATG4ARfIAQ_YAQHoAQH4AQKIAgGoAgO4AtzRkroGwAIB0gIkNzI2MWQ0OTEtNzExNS00MTA4LTk0N2MtYjkwMDZmMmZlYWU32AIF4AIB&aid=304142&lang=en-us&sb=1&src_elem=sb&src=hotel&dest_id=-290029&dest_type=city&group_adults=2&no_rooms=1&group_children=0','https://www.booking.com/searchresults.html?ss=Hurghada%2C+Egypt&ssne=Alexandria&ssne_untouched=Alexandria&highlighted_hotels=788831&efdco=1&label=gen173nr-1FCAEoggI46AdIM1gEaEOIAQGYATG4ARfIAQ_YAQHoAQH4AQKIAgGoAgO4AtzRkroGwAIB0gIkNzI2MWQ0OTEtNzExNS00MTA4LTk0N2MtYjkwMDZmMmZlYWU32AIF4AIB&aid=304142&lang=en-us&sb=1&src_elem=sb&src=hotel&dest_id=-290029&dest_type=city&group_adults=2&no_rooms=1&group_children=0','https://www.booking.com/searchresults.html?ss=Sharm+El+Sheikh%2C+Egypt&ssne=Cairo&ssne_untouched=Cairo&highlighted_hotels=788831&efdco=1&label=gen173nr-1FCAEoggI46AdIM1gEaEOIAQGYATG4ARfIAQ_YAQHoAQH4AQKIAgGoAgO4AtzRkroGwAIB0gIkNzI2MWQ0OTEtNzExNS00MTA4LTk0N2MtYjkwMDZmMmZlYWU32AIF4AIB&aid=304142&lang=en-us&sb=1&src_elem=sb&src=searchresults&dest_id=-302053&dest_type=city&group_adults=2&no_rooms=1&group_children=0','https://www.booking.com/searchresults.html?ss=Alexandria%2C+Egypt&ssne=Ain+Sokhna&ssne_untouched=Ain+Sokhna&highlighted_hotels=788831&efdco=1&label=gen173nr-1FCAEoggI46AdIM1gEaEOIAQGYATG4ARfIAQ_YAQHoAQH4AQKIAgGoAgO4AtzRkroGwAIB0gIkNzI2MWQ0OTEtNzExNS00MTA4LTk0N2MtYjkwMDZmMmZlYWU32AIF4AIB&aid=304142&lang=en-us&sb=1&src_elem=sb&src=searchresults&dest_id=-290263&dest_type=city&group_adults=2&no_rooms=1&group_children=0','https://www.booking.com/searchresults.html?ss=Cairo%2C+Egypt&ssne=Alexandria&ssne_untouched=Alexandria&highlighted_hotels=788831&efdco=1&label=gen173nr-1FCAEoggI46AdIM1gEaEOIAQGYATG4ARfIAQ_YAQHoAQH4AQKIAgGoAgO4AtzRkroGwAIB0gIkNzI2MWQ0OTEtNzExNS00MTA4LTk0N2MtYjkwMDZmMmZlYWU32AIF4AIB&aid=304142&lang=en-us&sb=1&src_elem=sb&src=searchresults&dest_id=-290692&dest_type=city&group_adults=2&no_rooms=1&group_children=0']
hotel_test=['https://www.booking.com/hotel/eg/the-grand-resort-hurghada.html?label=gen173nr-1FCAEoggI46AdIM1gEaEOIAQGYATG4ARfIAQ_YAQHoAQH4AQKIAgGoAgO4AtzRkroGwAIB0gIkNzI2MWQ0OTEtNzExNS00MTA4LTk0N2MtYjkwMDZmMmZlYWU32AIF4AIB&aid=304142&ucfs=1&arphpl=1&checkin=2024-11-26&checkout=2024-11-29&dest_id=-290029&dest_type=city&group_adults=2&req_adults=2&no_rooms=1&group_children=0&req_children=0&hpos=1&hapos=1&sr_order=popularity&srpvid=229b7882e20809ff&srepoch=1732556497&all_sr_blocks=71603801_376883182_2_85_0&highlighted_blocks=71603801_376883182_2_85_0&matching_block_id=71603801_376883182_2_85_0&sr_pri_blocks=71603801_376883182_2_85_0__21583&from_sustainable_property_sr=1&from=searchresults#tab-reviews','https://www.booking.com/hotel/eg/shellghada-blue-beach.html?label=gen173nr-1FCAEoggI46AdIM1gEaEOIAQGYATG4ARfIAQ_YAQHoAQH4AQKIAgGoAgO4AtzRkroGwAIB0gIkNzI2MWQ0OTEtNzExNS00MTA4LTk0N2MtYjkwMDZmMmZlYWU32AIF4AIB&aid=304142&ucfs=1&arphpl=1&dest_id=-290029&dest_type=city&group_adults=2&req_adults=2&no_rooms=1&group_children=0&req_children=0&hpos=2&hapos=2&sr_order=popularity&srpvid=6f6c7b5019bd0087&srepoch=1732555938&from=searchresults']
hotel_links=[]

def get_hotels_link_in_each_city(cities, Cities_url):
    print("Scraping hotels links in each city started")
    for url, city in zip(Cities_url, cities):
        try:
            response = requests.get(url, headers=HEADERS)
            print(city,"Connection success:",response.status_code)
            soup = bs(response.content, 'html.parser')
            
            for container in soup.find_all('div', {'data-testid': 'title'}):
                try:
                    hotel_name = container.text.strip()
                    parent = container.find_parent('div')
                    hotel_link = parent.find('a', href=True)
                    
                    if hotel_link:
                        hotel_links.append({'name': hotel_name, 'link': hotel_link['href']})
                except Exception as e:
                    print(f"Error extracting hotel in {city}: {e}")
        except Exception as e:
            print(f"Connection error for {city}: {e}")
        if(hotel_links == []):
            print("No hotels found")
        else:
            print("Hotels links scraped successfully")
    return hotel_links


   
def get_reviews(hotel_links):
    print("Scraping reviews started")
   # Initialize data containers
    reviewer_name, review_title, review_good_text, review_bad_text, hotel_name = [], [], [], [], []

    # Set up the Selenium driver (Firefox)
    driver = webdriver.Firefox(service=Service(GeckoDriverManager().install()))

    for hotel in hotel_links:
        hotelName = hotel['name']
        hotelLink = hotel['link']
        print("Scraping reviews for",hotelName,"started")
        try:
            # Open hotel page
            driver.get(hotelLink)

            # Wait for the reviews button to load and click it
            wait = WebDriverWait(driver, 10)
            reviews_button = wait.until(EC.element_to_be_clickable((By.XPATH, '//button[@data-testid="fr-read-all-reviews"]')))
            reviews_button.click()

            # Wait for review cards to load
            wait.until(EC.presence_of_all_elements_located((By.XPATH, '//div[@data-testid="review-card"]')))

            # Extract review cards
            reviews_cards = driver.find_elements(By.XPATH, '//div[@data-testid="review-card"]')
            for card in reviews_cards:
                # Extract hotel name
                hotel_name.append(hotelName)

                # Extract reviewer name
                try:
                    name = card.find_element(By.XPATH, './/div[@aria-label="Reviewer"]//div[@data-testid="review-avatar"] //div[@class="a3332d346a e6208ee469"]').text
                    print(name)
                except:
                    name = "No Name"
                reviewer_name.append(name)

                # Extract review title
                try:
                    title = card.find_element(By.XPATH, './/div[@aria-label="Review"]//h3[@data-testid="review-title"]').text
                    print(title)
                except:
                    title = "No Title"
                review_title.append(title)

                # Extract positive review text
                try:
                    good_text = card.find_element(By.XPATH, './/div[@aria-label="Review"]//div[@class="a53cbfa6de b5726afd0b"]').text
                    print(good_text)
                except:
                    good_text = "No Positive Review"
                review_good_text.append(good_text)

                # Extract negative review text
                try:
                    bad_text = card.find_element(By.XPATH, './/div[@aria-label="Review"]//div[@data-testid="review-negative-text"]//div[@class="a53cbfa6de b5726afd0b"]').text
                    print(bad_text)
                except:
                    bad_text = "No Negative Review"
                review_bad_text.append(bad_text)

        except Exception as e:
            print(f"Error processing hotel {hotel}: {e}")

    # Quit driver
    driver.quit()

    if(reviewer_name == []):
        print("No reviews scraped")
    else :    
        # Create a DataFrame to store the data
        reviews_df = pd.DataFrame({
            "Reviewer Name": reviewer_name,
            "Review Title": review_title,
            "Positive Review": review_good_text,
            "Negative Review": review_bad_text
        })

        # Save to a CSV file
        reviews_df.to_csv(r"C:\Users\Kariem\Desktop\Capstone_Porject\repo\Smart-Tourism-Dev-Sys\Data_collecting\hotel_reviews.csv", index=False)
        print("Data saved to hotel_reviews.csv")


if __name__ == "__main__":
    hotel_links = get_hotels_link_in_each_city(cities, Cities_url)
    get_reviews(hotel_links)

    #last update the script can scrap reviews 
    #   of the first hotel  
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.firefox import GeckoDriverManager  # Use this for Firefox
from time import sleep


# URLs to scrape
hotel_test = [
    'https://www.booking.com/hotel/eg/the-grand-resort-hurghada.html?label=gen173nr-1FCAEoggI46AdIM1gEaEOIAQGYATG4ARfIAQ_YAQHoAQH4AQKIAgGoAgO4AtzRkroGwAIB0gIkNzI2MWQ0OTEtNzExNS00MTA4LTk0N2MtYjkwMDZmMmZlYWU32AIF4AIB&aid=304142&ucfs=1&arphpl=1&checkin=2024-11-26&checkout=2024-11-29&dest_id=-290029&dest_type=city&group_adults=2&req_adults=2&no_rooms=1&group_children=0&req_children=0&hpos=1&hapos=1&sr_order=popularity&srpvid=229b7882e20809ff&srepoch=1732556497&all_sr_blocks=71603801_376883182_2_85_0&highlighted_blocks=71603801_376883182_2_85_0&matching_block_id=71603801_376883182_2_85_0&sr_pri_blocks=71603801_376883182_2_85_0__21583&from_sustainable_property_sr=1&from=searchresults#tab-reviews',
    'https://www.booking.com/hotel/eg/shellghada-blue-beach.html?label=gen173nr-1FCAEoggI46AdIM1gEaEOIAQGYATG4ARfIAQ_YAQHoAQH4AQKIAgGoAgO4AtzRkroGwAIB0gIkNzI2MWQ0OTEtNzExNS00MTA4LTk0N2MtYjkwMDZmMmZlYWU32AIF4AIB&aid=304142&ucfs=1&arphpl=1&dest_id=-290029&dest_type=city&group_adults=2&req_adults=2&no_rooms=1&group_children=0&req_children=0&hpos=2&hapos=2&sr_order=popularity&srpvid=6f6c7b5019bd0087&srepoch=1732555938&from=searchresults'
]

# Initialize data containers
reviewer_name, review_title, review_good_text, review_bad_text = [], [], [], []

# Set up the Selenium driver (Firefox)
driver = webdriver.Firefox(service=Service(GeckoDriverManager().install()))

for hotel in hotel_test:
    try:
        # Open hotel page
        driver.get(hotel)
        
        # Wait for the reviews button to load and click it
        wait = WebDriverWait(driver, 10)
        reviews_button = wait.until(EC.element_to_be_clickable((By.XPATH, '//button[@data-testid="read-all-actionable"]')))
        reviews_button.click()

        # Wait for review cards to load
        wait.until(EC.presence_of_all_elements_located((By.XPATH, '//div[@data-testid="review-card"]')))

        # Extract review cards
        reviews_cards = driver.find_elements(By.XPATH, '//div[@data-testid="review-card"]')
        for card in reviews_cards:
            # Extract reviewer name
            try:
                name = card.find_element(By.XPATH, './/div[@aria-label="Reviewer"]//div[@data-testid="review-avatar"] //div[@class="a3332d346a e6208ee469"]').text
            except:
                name = "No Name"
            reviewer_name.append(name)

            # Extract review title
            try:
                title = card.find_element(By.XPATH, './/div[@aria-label="Review"]//h3[@data-testid="review-title"]').text
            except:
                title = "No Title"
            review_title.append(title)

            # Extract positive review text
            try:
                good_text = card.find_element(By.XPATH, './/div[@aria-label="Review"]//div[@class="a53cbfa6de b5726afd0b"]').text
            except:
                good_text = "No Positive Review"
            review_good_text.append(good_text)

            # Extract negative review text
            try:
                bad_text = card.find_element(By.XPATH, './/div[@aria-label="Review"]//div[@data-testid="review-negative-text"]//div[@class="a53cbfa6de b5726afd0b"]').text
            except:
                bad_text = "No Negative Review"
            review_bad_text.append(bad_text)

    except Exception as e:
        print(f"Error processing hotel {hotel}: {e}")

# Quit driver
driver.quit()

# Check if any reviews were scraped
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

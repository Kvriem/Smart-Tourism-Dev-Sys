from bs4 import BeautifulSoup as bs
import requests
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.firefox import GeckoDriverManager 
from requests.exceptions import RequestException
import random
from time import sleep
from ReviewsScript import safe_request ,HEADERS_LIST
global activities_list

cities=dict(key='City_name', value='City_url')

cities = {
    'Sharm El Sheikh':'https://www.booking.com/attractions/searchresults/eg/sherm-el-sheikh.html?aid=304142&label=gen173nr-1FCAEoggI46AdIM1gEaEOIAQGYATG4ARfIAQ_YAQHoAQH4AQKIAgGoAgO4AvfAproGwAIB0gIkOGU2MDZhMDctYjFjOC00MTcyLTg0ZjgtMTA1ZWIxNDkyNjVj2AIF4AIB&source=search_box&sort_by=trending',
    'Luxor': 'https://www.booking.com/attractions/searchresults/eg/luxor.html?aid=304142&label=gen173nr-1FCAEoggI46AdIM1gEaEOIAQGYATG4ARfIAQ_YAQHoAQH4AQKIAgGoAgO4AvfAproGwAIB0gIkOGU2MDZhMDctYjFjOC00MTcyLTg0ZjgtMTA1ZWIxNDkyNjVj2AIF4AIB&source=search_box',
    'Hurghada':'https://www.booking.com/attractions/searchresults/eg/al-ghardaqah.html?aid=304142&label=gen173nr-1FCAEoggI46AdIM1gEaEOIAQGYATG4ARfIAQ_YAQHoAQH4AQKIAgGoAgO4AvfAproGwAIB0gIkOGU2MDZhMDctYjFjOC00MTcyLTg0ZjgtMTA1ZWIxNDkyNjVj2AIF4AIB&source=search_box',
    'Cairo':'https://www.booking.com/attractions/searchresults/eg/cairo.html?aid=304142&label=gen173nr-1FCAEoggI46AdIM1gEaEOIAQGYATG4ARfIAQ_YAQHoAQH4AQKIAgGoAgO4AvfAproGwAIB0gIkOGU2MDZhMDctYjFjOC00MTcyLTg0ZjgtMTA1ZWIxNDkyNjVj2AIF4AIB&source=search_box',
    'Alexandria':'https://www.booking.com/attractions/searchresults/eg/alexandria.html?aid=304142&label=gen173nr-1FCAEoggI46AdIM1gEaEOIAQGYATG4ARfIAQ_YAQHoAQH4AQKIAgGoAgO4AvfAproGwAIB0gIkOGU2MDZhMDctYjFjOC00MTcyLTg0ZjgtMTA1ZWIxNDkyNjVj2AIF4AIB&source=search_box'
}

#test=dict(key='City_name', value='City_url')
#test=['https://www.booking.com/attractions/searchresults/eg/sherm-el-sheikh.html?aid=304142&label=gen173nr-1FCAEoggI46AdIM1gEaEOIAQGYATG4ARfIAQ_YAQHoAQH4AQKIAgGoAgO4AvfAproGwAIB0gIkOGU2MDZhMDctYjFjOC00MTcyLTg0ZjgtMTA1ZWIxNDkyNjVj2AIF4AIB&source=search_box&sort_by=trending']

total_requests = 0
selenium_requests = 0
driver = webdriver.Firefox(service=Service(GeckoDriverManager().install()))

activities_list=[]

def get_activities_in_each_city(cities):
    print('Start Scraping Activities')
    city_link=cities.values()
    city_name=cities.keys()
    for link,name in zip(city_link,city_name):          
            driver.get(link)
            sleep(3)
        #getting all activities in each city 
            try:
                activites =driver.find_elements(By.XPATH, '//a[@data-link-type="title"]')
                for activity in activites:
                    activity_link = activity.get_attribute('href')  # Extract the link
                    activities_list.append({'name': name, 'link': activity_link})
            except:
                print('Error in getting activities')
            
    return activities_list


def get_activites_reviews (activities_list):
    print('Start Scraping Reviews for each activity')
    headers_written = False
    
    iteration = 0
    for activity in activities_list:
            name=activity['name']
            activity=activity['link']
            clicked = False

            review_text , reviewer_nationality, reviewer_name, review_date, travel_type,city_name =[], [],[],[],[],[]
            print(f'start scraping activities of: {name}')
            driver.get(activity)
            sleep(3)
            # show all reviews 
            try:
                driver.find_element(By.XPATH, '//button[@class="a83ed08757 f88a5204c2 css-15bxs5u b98133fb50"]').click()
                clicked = True  
                sleep(3)
            except:
                print('Error in showing all reviews')
            
            # get reviews
            if clicked:

                reviews_card= driver.find_elements(By.XPATH,'.//li[@class="a8b57ad3ff fd727d5f06"]')
                for review in reviews_card:
                    city_name .append(name)
                    # get reviewer name
                    try:
                        reviewer = review.find_element(By.XPATH,'.//div[@class="css-1lxwves"]//div[@class="a3332d346a"]').text
                    except:
                        reviewer = "NULL"
                    reviewer_name.append(reviewer)

                    # get reviewer nationality
                    try:
                        nationality = review.find_element(By.XPATH,'.//div[@class="css-1lxwves"]//div[@class="abf093bdfe f45d8e4c32"]').text
                    except:
                        nationality = "NULL"
                    reviewer_nationality.append(nationality)
                    
                    # get review text
                    try:
                        review_= review.find_element(By.XPATH,'.//div[@class="a53cbfa6de fd0c3f4521 fc409351f3 e8bb16538d"]').text
                    except:
                        review_ = "NULL"
                    review_text.append(review_)

                    # get review date and travel type
                    try:
                        type_ = review.find_element(By.XPATH, './/div[@class="css-1ltm3ye"]//div[@class="abf093bdfe f45d8e4c32"]')
                        type_ = type_.text
                    except:
                        type_ = "NULL"
                    travel_type.append(type_)

                    # get reviewer travel type
                    try:
                        date_ = review.find_element(By.XPATH, './/div[@class="abf093bdfe f45d8e4c32"][contains(text(), "Posted")]')
                        date_ = date_.text
                    except:
                        date_ = "NULL"

                    # Append to respective lists
                    review_date.append(date_)
                    
                    
                # create a dataframe for reviews    
                reviews_df = pd.DataFrame({'city_name':city_name
                                            ,'reviewer_name':reviewer_name
                                            ,'reviewer_nationality' : reviewer_nationality
                                            ,'review_text':review_text
                                            ,'review_date':review_date
                                            ,'travel_type':travel_type
                                            })
                
                # save reviews to csv
                output_file = 'activities_reviews.csv'

                reviews_df.to_csv(output_file, mode='a',index=False, header=not headers_written)
                headers_written = True  # Set flag to skip headers for subsequent hotels
                print(f"Appended reviews to {output_file}")
                    
            else:
                print('Error in clicking reviews button')

            iteration += 1
        
    return reviews_df

get_activites_reviews(get_activities_in_each_city(cities))

driver.quit()

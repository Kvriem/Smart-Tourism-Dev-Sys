#!/usr/bin/env python3
"""
Test script to verify Firefox functionality for the web scraping project.
"""
import os
import sys
import logging
from selenium import webdriver
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.firefox import GeckoDriverManager

def setup_logging():
    """Setup basic logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def test_firefox_installation():
    """Test if Firefox is properly installed"""
    try:
        import subprocess
        result = subprocess.run(['firefox', '--version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            logging.info(f"Firefox version: {result.stdout.strip()}")
            return True
        else:
            logging.error("Firefox not found in system PATH")
            return False
    except Exception as e:
        logging.error(f"Error checking Firefox installation: {e}")
        return False

def test_geckodriver():
    """Test geckodriver functionality"""
    try:
        # Test webdriver_manager
        logging.info("Testing webdriver_manager...")
        driver_path = GeckoDriverManager().install()
        logging.info(f"GeckoDriverManager successful: {driver_path}")
        return driver_path
    except Exception as e:
        logging.warning(f"GeckoDriverManager failed: {e}")
        
        # Try system PATH
        import shutil
        system_driver = shutil.which('geckodriver')
        if system_driver:
            logging.info(f"Found geckodriver in system PATH: {system_driver}")
            return system_driver
        
        logging.error("No geckodriver found")
        return None

def test_basic_selenium():
    """Test basic Selenium functionality with Firefox"""
    driver = None
    try:
        # Setup Firefox options
        options = Options()
        options.add_argument('--headless')  # Run in headless mode for testing
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        
        # Get geckodriver
        driver_path = test_geckodriver()
        if not driver_path:
            return False
        
        # Setup service
        service = Service(driver_path)
        
        # Create driver
        logging.info("Creating Firefox WebDriver...")
        driver = webdriver.Firefox(service=service, options=options)
        driver.implicitly_wait(10)
        driver.set_page_load_timeout(30)
        
        # Test basic navigation
        logging.info("Testing navigation to Google...")
        driver.get("https://www.google.com")
        
        # Check if page loaded
        title = driver.title
        logging.info(f"Page title: {title}")
        
        if "google" in title.lower():
            logging.info("✅ Firefox WebDriver test successful!")
            return True
        else:
            logging.error("❌ Page did not load correctly")
            return False
            
    except Exception as e:
        logging.error(f"❌ Firefox WebDriver test failed: {e}")
        return False
    finally:
        if driver:
            try:
                driver.quit()
                logging.info("WebDriver closed successfully")
            except:
                pass

def test_booking_com_access():
    """Test accessing Booking.com (the target site)"""
    driver = None
    try:
        # Setup Firefox options for real scraping
        options = Options()
        options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        
        # Anti-detection preferences
        options.set_preference('dom.webdriver.enabled', False)
        options.set_preference('useAutomationExtension', False)
        options.set_preference("general.useragent.override", 
                             "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
        
        # Get geckodriver
        driver_path = test_geckodriver()
        if not driver_path:
            return False
        
        service = Service(driver_path)
        driver = webdriver.Firefox(service=service, options=options)
        driver.implicitly_wait(10)
        driver.set_page_load_timeout(30)
        
        # Test Booking.com access
        logging.info("Testing access to Booking.com...")
        driver.get("https://www.booking.com")
        
        # Wait for page to load and check title
        WebDriverWait(driver, 15).until(
            lambda d: d.execute_script("return document.readyState") == "complete"
        )
        
        title = driver.title
        logging.info(f"Booking.com page title: {title}")
        
        if "booking" in title.lower():
            logging.info("✅ Booking.com access test successful!")
            return True
        else:
            logging.warning("⚠️ Booking.com page may be blocked or different")
            return False
            
    except Exception as e:
        logging.error(f"❌ Booking.com access test failed: {e}")
        return False
    finally:
        if driver:
            try:
                driver.quit()
            except:
                pass

def main():
    """Run all tests"""
    setup_logging()
    
    logging.info("🚀 Starting Firefox compatibility tests...")
    
    # Test 1: Firefox installation
    logging.info("\n" + "="*50)
    logging.info("TEST 1: Firefox Installation")
    logging.info("="*50)
    if not test_firefox_installation():
        logging.error("❌ Firefox installation test failed!")
        return False
    
    # Test 2: Geckodriver
    logging.info("\n" + "="*50)
    logging.info("TEST 2: Geckodriver Setup")
    logging.info("="*50)
    if not test_geckodriver():
        logging.error("❌ Geckodriver test failed!")
        return False
    
    # Test 3: Basic Selenium
    logging.info("\n" + "="*50)
    logging.info("TEST 3: Basic Selenium WebDriver")
    logging.info("="*50)
    if not test_basic_selenium():
        logging.error("❌ Basic Selenium test failed!")
        return False
    
    # Test 4: Booking.com access
    logging.info("\n" + "="*50)
    logging.info("TEST 4: Booking.com Access")
    logging.info("="*50)
    if not test_booking_com_access():
        logging.warning("⚠️ Booking.com access test had issues, but Firefox works")
    
    logging.info("\n" + "="*50)
    logging.info("🎉 ALL FIREFOX TESTS COMPLETED!")
    logging.info("✅ Firefox is ready for your scraping script!")
    logging.info("="*50)
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
import logging
import sys
import os

# Add the src directory to Python path to import our modules
sys.path.append('/home/kariem/airflow/src')

# Import configuration utility
from config.scraping_config import get_runtime_config, config_manager, load_preset

# Import functions from hotel_scraper.py
from scrapers.hotel_scraper import (
    initialize_logging,
    create_db_connection,
    create_reviews_table,
    get_hotel_links,
    setup_driver,
    scrape_hotel_reviews,
    mark_hotel_as_scraped,
    get_random_user_agent,
    main_scraping_process  # Add this import
)

# Try to import email operator, fallback if not available
try:
    from airflow.providers.smtp.operators.email import EmailOperator
    EMAIL_AVAILABLE = True
except ImportError:
    try:
        from airflow.operators.email import EmailOperator
        EMAIL_AVAILABLE = True
    except ImportError:
        logging.warning("Email operator not available, email notifications will be skipped")
        EMAIL_AVAILABLE = False

# Default arguments for the DAG
default_args = {
    'owner': 'kariem',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,  # Disable built-in email since we'll handle it manually
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=30),
    'catchup': False
}

# Create the DAG
dag = DAG(
    'quarterly_hotel_reviews_scraper_checkpoint1',  # Different DAG name for checkpoint
    default_args=default_args,
    description='Quarterly hotel reviews scraping from Booking.com - Checkpoint 1',
    schedule_interval='0 0 1 */3 *',  # First day of every quarter at midnight
    max_active_runs=1,
    tags=['scraping', 'hotels', 'quarterly', 'checkpoint'],
    max_active_tasks=2,  # Limit concurrent tasks
    catchup=False
)

def task_initialize_logging(**context):
    """Task to initialize logging system"""
    try:
        initialize_logging()
        logging.info("Logging initialized successfully for quarterly scraping")
        return "Logging initialized"
    except Exception as e:
        logging.error(f"Failed to initialize logging: {e}")
        raise

def task_setup_database(**context):
    """Task to setup database connection and create tables"""
    try:
        connection = create_db_connection()
        if not connection:
            raise Exception("Failed to create database connection")
        
        create_reviews_table(connection)
        
        # Store connection info in XCom for other tasks
        connection.close()
        logging.info("Database setup completed successfully")
        return "Database setup completed"
    except Exception as e:
        logging.error(f"Database setup failed: {e}")
        raise

def task_get_hotel_links(**context):
    """Task to fetch hotel links from database"""
    try:
        connection = create_db_connection()
        if not connection:
            raise Exception("Failed to create database connection")
        
        hotels = get_hotel_links(connection)
        connection.close()
        
        if not hotels:
            raise Exception("No hotel links found in database")
        
        logging.info(f"Retrieved {len(hotels)} hotel links")
        
        # Store hotel count in XCom for monitoring
        context['task_instance'].xcom_push(key='hotel_count', value=len(hotels))
        return f"Retrieved {len(hotels)} hotels"
    except Exception as e:
        logging.error(f"Failed to get hotel links: {e}")
        raise

def task_setup_webdriver(**context):
    """Task to setup and test WebDriver"""
    try:
        # Load quarterly configuration
        load_preset("quarterly")
        
        driver = setup_driver()
        if not driver:
            raise Exception("Failed to setup WebDriver")
        
        # Test basic functionality
        driver.get("https://www.google.com")
        title = driver.title
        driver.quit()
        
        if "google" not in title.lower():
            raise Exception("WebDriver test failed")
        
        logging.info("WebDriver setup and test completed successfully")
        return "WebDriver ready"
    except Exception as e:
        logging.error(f"WebDriver setup failed: {e}")
        raise

def task_scrape_hotel_reviews(**context):
    """Main task to scrape hotel reviews using the main_scraping_process function"""
    try:
        # Get quarterly configuration
        runtime_config = get_runtime_config("quarterly")
        max_hotels = runtime_config["max_hotels"]
        max_pages = runtime_config["max_pages_per_hotel"]
        
        logging.info(f"Using quarterly config - Max hotels: {max_hotels}, Max pages: {max_pages}")
        
        # Load quarterly preset configuration
        load_preset("quarterly")
        
        # Store initial values for reporting
        initial_hotel_count = 0
        try:
            connection = create_db_connection()
            if connection:
                hotels = get_hotel_links(connection)
                initial_hotel_count = len(hotels)
                connection.close()
        except:
            pass
        
        # Run the main scraping process with configuration limits
        logging.info("Starting main scraping process...")
        main_scraping_process(
            max_hotels=max_hotels,
            max_pages_per_hotel=max_pages
        )
        
        # Get final results for reporting
        try:
            connection = create_db_connection()
            if connection:
                cursor = connection.cursor()
                
                # Count total reviews scraped today
                cursor.execute("""
                    SELECT COUNT(*) FROM bronze.hotels_reviews_test 
                    WHERE scraped_at::date = CURRENT_DATE
                """)
                total_reviews_today = cursor.fetchone()[0] or 0
                
                # Count hotels processed today (marked as scraped)
                cursor.execute("""
                    SELECT COUNT(*) FROM bronze.hotel_links 
                    WHERE is_scraped = TRUE
                """)
                total_scraped_hotels = cursor.fetchone()[0] or 0
                
                cursor.close()
                connection.close()
                
                # Calculate processed hotels (approximate)
                hotels_processed = min(total_scraped_hotels, max_hotels) if max_hotels else total_scraped_hotels
                hotels_failed = 0  # Will be calculated based on logs if needed
                
                # Store results in XCom
                results = {
                    'total_reviews': total_reviews_today,
                    'hotels_processed': hotels_processed,
                    'hotels_failed': hotels_failed,
                    'config_used': runtime_config
                }
                
                for key, value in results.items():
                    context['task_instance'].xcom_push(key=key, value=value)
                
                logging.info(f"Scraping completed. Reviews today: {total_reviews_today}, "
                            f"Hotels processed: {hotels_processed}")
                
                return f"Scraped {total_reviews_today} reviews from {hotels_processed} hotels"
                
        except Exception as e:
            logging.warning(f"Could not get final statistics: {e}")
            # Fallback return
            return "Scraping process completed (statistics unavailable)"
        
    except Exception as e:
        logging.error(f"Scraping task failed: {e}")
        raise

def task_generate_report(**context):
    """Task to generate scraping report"""
    try:
        # Get results from previous task
        ti = context['task_instance']
        total_reviews = ti.xcom_pull(task_ids='scrape_reviews', key='total_reviews') or 0
        hotels_processed = ti.xcom_pull(task_ids='scrape_reviews', key='hotels_processed') or 0
        hotels_failed = ti.xcom_pull(task_ids='scrape_reviews', key='hotels_failed') or 0
        hotel_count = ti.xcom_pull(task_ids='get_hotel_links', key='hotel_count') or 0
        
        # Calculate success rate
        success_rate = ((hotels_processed - hotels_failed) / hotels_processed * 100) if hotels_processed > 0 else 0
        
        report = f"""
        Quarterly Hotel Reviews Scraping Report
        =====================================
        
        Execution Date: {context['ds']}
        
        Results:
        - Total hotels in database: {hotel_count}
        - Hotels processed: {hotels_processed}
        - Hotels failed: {hotels_failed}
        - Success rate: {success_rate:.1f}%
        - Total reviews scraped: {total_reviews}
        
        Status: {'SUCCESS' if hotels_failed == 0 else 'PARTIAL SUCCESS' if total_reviews > 0 else 'FAILED'}
        """
        
        logging.info(report)
        
        # Store report in XCom
        context['task_instance'].xcom_push(key='report', value=report)
        
        return report
        
    except Exception as e:
        logging.error(f"Report generation failed: {e}")
        raise

def task_cleanup(**context):
    """Task to perform cleanup operations"""
    try:
        # Clean up any temporary files, logs older than 30 days, etc.
        import glob
        import os
        from datetime import datetime, timedelta
        
        # Clean old log files
        log_dir = '/home/kariem/airflow/dags'
        old_logs = glob.glob(os.path.join(log_dir, 'scraping_log*.log'))
        
        cutoff_date = datetime.now() - timedelta(days=30)
        
        for log_file in old_logs:
            try:
                if os.path.getmtime(log_file) < cutoff_date.timestamp():
                    os.remove(log_file)
                    logging.info(f"Removed old log file: {log_file}")
            except:
                pass
        
        logging.info("Cleanup completed")
        return "Cleanup completed"
        
    except Exception as e:
        logging.error(f"Cleanup failed: {e}")
        # Don't raise exception for cleanup failures
        return f"Cleanup failed: {e}"

def task_send_notification(**context):
    """Task to send notification (fallback if email operator not available)"""
    try:
        # Get report from previous task
        ti = context['task_instance']
        report = ti.xcom_pull(task_ids='generate_report', key='report')
        
        # If email operator is not available, just log the report
        if not EMAIL_AVAILABLE:
            logging.info("EMAIL NOTIFICATION (Email operator not available):")
            logging.info("=" * 50)
            logging.info(report)
            logging.info("=" * 50)
            return "Notification logged (email not configured)"
        
        # If we reach here, email should be handled by the EmailOperator task
        return "Email notification handled by EmailOperator"
        
    except Exception as e:
        logging.error(f"Notification task failed: {e}")
        return f"Notification failed: {e}"

# Define tasks
init_logging_task = PythonOperator(
    task_id='initialize_logging',
    python_callable=task_initialize_logging,
    dag=dag
)

setup_db_task = PythonOperator(
    task_id='setup_database',
    python_callable=task_setup_database,
    dag=dag
)

get_links_task = PythonOperator(
    task_id='get_hotel_links',
    python_callable=task_get_hotel_links,
    dag=dag
)

setup_driver_task = PythonOperator(
    task_id='setup_webdriver',
    python_callable=task_setup_webdriver,
    dag=dag
)

scrape_task = PythonOperator(
    task_id='scrape_reviews',
    python_callable=task_scrape_hotel_reviews,
    dag=dag,
    execution_timeout=timedelta(hours=4),  # Reduced from 6 hours
    retries=1,  # Override default retries for this heavy task
    retry_delay=timedelta(minutes=15),  # Shorter retry delay
    max_active_tis_per_dag=1  # Ensure only one instance runs at a time
)

report_task = PythonOperator(
    task_id='generate_report',
    python_callable=task_generate_report,
    dag=dag
)

cleanup_task = PythonOperator(
    task_id='cleanup',
    python_callable=task_cleanup,
    dag=dag,
    trigger_rule='all_done'  # Run even if previous tasks failed
)

notification_task = PythonOperator(
    task_id='send_notification',
    python_callable=task_send_notification,
    dag=dag,
    trigger_rule='all_done'
)

# Define task dependencies
init_logging_task >> setup_db_task >> get_links_task >> setup_driver_task >> scrape_task >> report_task >> [cleanup_task, notification_task]

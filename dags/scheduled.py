from airflow import DAG
from datetime import datetime, timedelta
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from airflow.utils.task_group import TaskGroup
import logging
import sys
import os
import time
import pandas as pd

# Add the src directory to Python path to import our modules
sys.path.append('/home/kariem/airflow/src')

# Import configuration utility
from config.scraping_config import config_manager, load_preset

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
    get_runtime_config,  # Import this from hotel_scraper instead
    main_scraping_process  # Add this import
)

# Import preprocessing functions
from nlp.preprocessing_nlp_ref.preprocessing_script import (
    create_db_engine,
    ensure_schema_and_table_exist,
    read_from_bronze_table,
    process_reviews_data_from_db,
    ingest_to_silver_table
)

# Import translation functions
from nlp.preprocessing_nlp_ref.translation_local_gpu import (
    create_optimized_engine,
    read_data_optimized,
    ensure_table_exists_optimized,
    translate_reviews_ultra_fast,
    ingest_data_ultra_fast,
    PerformanceCache,
    perf_cache
)

# Import NLP processing pipeline functions
from nlp.preprocessing_nlp_ref.nlp_processing_pipeline import (
    setup_libraries,
    connect_and_fetch_data,
    preprocess_data,
    extract_semantic_phrases_batch,
    extract_and_clean_tokens,
    filter_tourism_tokens,
    extract_tourism_tokens_per_row,
    prepare_final_dataset,
    ingest_to_database,
    TOURISM_VOCABULARY
)


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
    'quarterly_hotel_reviews_scraper',
    default_args=default_args,
    description='Quarterly hotel reviews scraping from Booking.com',
    schedule_interval='0 0 1 */3 *',  # First day of every quarter at midnight
    max_active_runs=1,
    tags=['scraping', 'hotels', 'quarterly'],
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
        runtime_config = get_runtime_config("test")
        max_hotels = runtime_config["max_hotels"]
        max_pages = runtime_config["max_pages_per_hotel"]
        
        logging.info(f"Using test config - Max hotels: {max_hotels}, Max pages: {max_pages}")
        
        # Load quarterly preset configuration
        load_preset("test")
        
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

def task_truncate_reviews_table(**context):
    """Task to truncate the reviews table before scraping"""
    try:
        connection = create_db_connection()
        if not connection:
            raise Exception("Failed to create database connection")
        
        cursor = connection.cursor()
        cursor.execute("TRUNCATE TABLE bronze.hotels_reviews_test")
        connection.commit()
        cursor.close()
        connection.close()
        
        logging.info("Reviews table truncated successfully")
        return "Reviews table truncated"
    except Exception as e:
        logging.error(f"Failed to truncate reviews table: {e}")
        raise

def task_preprocessing_setup(**context):
    """Task to setup preprocessing environment"""
    try:
        engine = create_db_engine()
        ensure_schema_and_table_exist(engine)
        engine.dispose()
        
        logging.info("Preprocessing setup completed successfully")
        return "Preprocessing setup completed"
    except Exception as e:
        logging.error(f"Preprocessing setup failed: {e}")
        raise

def task_data_preprocessing(**context):
    """Task to process data from bronze to silver layer"""
    try:
        engine = create_db_engine()
        
        # Process reviews data
        processed_df = process_reviews_data_from_db(engine)
        
        # Ingest to silver table
        ingest_to_silver_table(processed_df, engine)
        
        engine.dispose()
        
        # Store results in XCom
        context['task_instance'].xcom_push(key='processed_records', value=len(processed_df))
        
        logging.info(f"Data preprocessing completed. Processed {len(processed_df)} records")
        return f"Processed {len(processed_df)} records to silver layer"
    except Exception as e:
        logging.error(f"Data preprocessing failed: {e}")
        raise

def task_preprocessing_validation(**context):
    """Task to validate preprocessing results"""
    try:
        engine = create_db_engine()
        
        # Get count of records in silver table
        with engine.connect() as conn:
            from sqlalchemy import text
            result = conn.execute(text("SELECT COUNT(*) FROM silver.reviews_cleaned"))
            silver_count = result.fetchone()[0]
            
            # Get count of records in bronze table
            result = conn.execute(text("SELECT COUNT(*) FROM bronze.hotels_reviews_test"))
            bronze_count = result.fetchone()[0]
        
        engine.dispose()
        
        # Store validation results
        context['task_instance'].xcom_push(key='silver_count', value=silver_count)
        context['task_instance'].xcom_push(key='bronze_count', value=bronze_count)
        
        logging.info(f"Validation completed. Bronze: {bronze_count}, Silver: {silver_count} records")
        return f"Validation completed. Silver layer has {silver_count} records"
    except Exception as e:
        logging.error(f"Preprocessing validation failed: {e}")
        raise

def task_translation_setup(**context):
    """Task to setup translation environment and optimized database connection"""
    try:
        engine = create_optimized_engine()
        ensure_table_exists_optimized(engine)
        engine.dispose()
        
        logging.info("Translation setup completed successfully")
        return "Translation setup completed"
    except Exception as e:
        logging.error(f"Translation setup failed: {e}")
        raise

def task_load_data_for_translation(**context):
    """Task to load data from silver.reviews_cleaned for translation"""
    try:
        logging.info("Starting data loading for translation...")
        start_time = time.time()
        
        # Create optimized engine
        engine = create_optimized_engine()
        
        # Query to load data from silver.reviews_cleaned (matching preprocessing script schema)
        query = """
        SELECT * FROM silver.reviews_cleaned
        """
        
        try:
            # Use direct psycopg2 connection like in preprocessing script
            import psycopg2
            from urllib.parse import urlparse
            
            # Parse the connection string (assuming same DB as preprocessing)
            DB_CONNECTION_STRING = "postgresql://neondb_owner:npg_ExFXHY8yiNT0@ep-lingering-term-ab7pbfql-pooler.eu-west-2.aws.neon.tech/neondb?sslmode=require"
            parsed = urlparse(DB_CONNECTION_STRING)
            
            # Create psycopg2 connection
            conn = psycopg2.connect(
                host=parsed.hostname,
                port=parsed.port or 5432,
                database=parsed.path[1:],  # Remove leading '/'
                user=parsed.username,
                password=parsed.password,
                sslmode='require'
            )
            
            df = pd.read_sql(query, conn)
            conn.close()
            
            elapsed_time = time.time() - start_time
            logging.info(f"Successfully loaded {len(df)} records from silver.reviews_cleaned in {elapsed_time:.2f} seconds")
            
            # Log data quality metrics like preprocessing script
            if len(df) > 0:
                non_null_positive = df['positive_review'].notna().sum()
                non_null_negative = df['negative_review'].notna().sum()
                sentiment_distribution = df['sentiment_classification'].value_counts().sort_index()
                
                logging.info(f"Data quality metrics:")
                logging.info(f"  Records with positive reviews: {non_null_positive}")
                logging.info(f"  Records with negative reviews: {non_null_negative}")
                logging.info(f"  Sentiment distribution: {sentiment_distribution.to_dict()}")
                logging.info(f"  Dataset shape: {df.shape}")
                logging.info(f"  Columns: {list(df.columns)}")
            
        except Exception as e:
            logging.error(f"Error reading from silver.reviews_cleaned: {e}")
            raise
        finally:
            engine.dispose()
        
        # Store data info in XCom (enhanced metrics)
        context['task_instance'].xcom_push(key='translation_records_count', value=len(df))
        context['task_instance'].xcom_push(key='translation_data_shape', value=str(df.shape))
        context['task_instance'].xcom_push(key='records_with_positive', value=int(non_null_positive) if len(df) > 0 else 0)
        context['task_instance'].xcom_push(key='records_with_negative', value=int(non_null_negative) if len(df) > 0 else 0)
        context['task_instance'].xcom_push(key='loading_time_seconds', value=round(elapsed_time, 2))
        
        logging.info(f"Data loading completed successfully")
        logging.info(f"Average loading rate: {len(df)/elapsed_time:.2f} records/second")
        
        return f"Loaded {len(df)} records for translation with shape {df.shape}"
        
    except Exception as e:
        logging.error(f"Failed to load data for translation: {e}")
        raise

def task_translate_reviews(**context):
    """Task to perform ultra-fast parallel translation of reviews"""
    try:
        engine = create_optimized_engine()
        
        # Read data for translation
        df = read_data_optimized(engine)
        
        # Perform translation
        translated_df = translate_reviews_ultra_fast(df)
        
        engine.dispose()
        
        # Store translation results in XCom
        context['task_instance'].xcom_push(key='translated_records_count', value=len(translated_df))
        context['task_instance'].xcom_push(key='translated_data_shape', value=str(translated_df.shape))
        
        # Get cache stats
        stats = perf_cache.get_stats()
        context['task_instance'].xcom_push(key='cache_hit_rate', value=stats['hit_rate'])
        context['task_instance'].xcom_push(key='cache_size', value=stats['cache_size'])
        
        logging.info(f"Translation completed for {len(translated_df)} records")
        logging.info(f"Cache hit rate: {stats['hit_rate']:.1f}%")
        return f"Translated {len(translated_df)} records with {stats['hit_rate']:.1f}% cache hit rate"
    except Exception as e:
        logging.error(f"Translation task failed: {e}")
        raise

def task_ingest_translated_data(**context):
    """Task to ingest translated data into silver.silver_translated table"""
    try:
        engine = create_optimized_engine()
        
        # Read data and translate (we need the translated dataframe)
        df = read_data_optimized(engine)
        translated_df = translate_reviews_ultra_fast(df)
        
        # Ingest translated data
        ingest_data_ultra_fast(translated_df, engine)
        
        engine.dispose()
        
        # Store ingestion results
        context['task_instance'].xcom_push(key='ingested_records_count', value=len(translated_df))
        
        logging.info(f"Successfully ingested {len(translated_df)} translated records")
        return f"Ingested {len(translated_df)} translated records"
    except Exception as e:
        logging.error(f"Translation data ingestion failed: {e}")
        raise

def task_translation_validation(**context):
    """Task to validate translation results"""
    try:
        engine = create_optimized_engine()
        
        # Get count of records in translated table
        with engine.connect() as conn:
            from sqlalchemy import text
            result = conn.execute(text("SELECT COUNT(*) FROM silver.silver_translated"))
            translated_count = result.fetchone()[0]
            
            # Get count of records with translated content
            result = conn.execute(text("""
                SELECT COUNT(*) FROM silver.silver_translated 
                WHERE negative_review_translated IS NOT NULL 
                   OR positive_review_translated IS NOT NULL
            """))
            records_with_translation = result.fetchone()[0]
            
            # Get count of source records
            result = conn.execute(text("SELECT COUNT(*) FROM silver.reviews_cleaned"))
            source_count = result.fetchone()[0]
        
        engine.dispose()
        
        # Store validation results
        context['task_instance'].xcom_push(key='translated_table_count', value=translated_count)
        context['task_instance'].xcom_push(key='records_with_translation', value=records_with_translation)
        context['task_instance'].xcom_push(key='source_count', value=source_count)
        
        logging.info(f"Translation validation completed:")
        logging.info(f"  Source records: {source_count}")
        logging.info(f"  Translated table records: {translated_count}")
        logging.info(f"  Records with translations: {records_with_translation}")
        
        return f"Validation: {translated_count} records in translated table, {records_with_translation} with translations"
    except Exception as e:
        logging.error(f"Translation validation failed: {e}")
        raise

def task_translation_report(**context):
    """Task to generate comprehensive translation report"""
    try:
        # Get results from previous tasks
        ti = context['task_instance']
        source_count = ti.xcom_pull(task_ids='translation_validation', key='source_count') or 0
        translated_count = ti.xcom_pull(task_ids='translation_validation', key='translated_table_count') or 0
        records_with_translation = ti.xcom_pull(task_ids='translation_validation', key='records_with_translation') or 0
        cache_hit_rate = ti.xcom_pull(task_ids='translate_reviews', key='cache_hit_rate') or 0
        cache_size = ti.xcom_pull(task_ids='translate_reviews', key='cache_size') or 0
        
        # Calculate translation coverage
        translation_coverage = (records_with_translation / translated_count * 100) if translated_count > 0 else 0
        
        report = f"""
        Translation Pipeline Report
        ==========================
        
        Execution Date: {context['ds']}
        
        Data Flow:
        - Source records (silver.reviews_cleaned): {source_count:,}
        - Translated records (silver.silver_translated): {translated_count:,}
        - Records with translations: {records_with_translation:,}
        - Translation coverage: {translation_coverage:.1f}%
        
        Performance:
        - Cache hit rate: {cache_hit_rate:.1f}%
        - Cache size: {cache_size:,} items
        
        Status: {'SUCCESS' if translated_count > 0 else 'FAILED'}
        """
        
        logging.info(report)
        
        # Store report in XCom
        context['task_instance'].xcom_push(key='translation_report', value=report)
        
        return report
        
    except Exception as e:
        logging.error(f"Translation report generation failed: {e}")
        raise

def task_setup_nlp_libraries(**context):
    """Task to setup NLP libraries"""
    try:
        libraries = setup_libraries()
        
        # Store library info in XCom
        context['task_instance'].xcom_push(key='libraries_setup', value=True)
        context['task_instance'].xcom_push(key='torch_available', value=libraries.get('torch_available', False))
        context['task_instance'].xcom_push(key='keybert_available', value=libraries.get('keybert_available', False))
        context['task_instance'].xcom_push(key='device', value=libraries.get('device', 'cpu'))
        
        logging.info("NLP libraries setup completed successfully")
        return "NLP libraries setup completed"
    except Exception as e:
        logging.error(f"NLP libraries setup failed: {e}")
        raise

def task_fetch_nlp_data(**context):
    """Task to fetch data for NLP processing"""
    try:
        df = connect_and_fetch_data()
        
        if df.empty:
            raise Exception("No data fetched from database")
        
        # Store data info in XCom
        context['task_instance'].xcom_push(key='nlp_data_count', value=len(df))
        context['task_instance'].xcom_push(key='nlp_data_shape', value=str(df.shape))
        
        logging.info(f"NLP data fetched successfully: {df.shape}")
        return f"Fetched {len(df)} records for NLP processing"
    except Exception as e:
        logging.error(f"NLP data fetching failed: {e}")
        raise

def task_preprocess_nlp_data(**context):
    """Task to preprocess data for NLP"""
    try:
        df = connect_and_fetch_data()
        df = preprocess_data(df)
        
        # Store preprocessing results
        context['task_instance'].xcom_push(key='preprocessed_data_count', value=len(df))
        
        logging.info(f"NLP data preprocessing completed for {len(df)} records")
        return f"Preprocessed {len(df)} records"
    except Exception as e:
        logging.error(f"NLP data preprocessing failed: {e}")
        raise

def task_extract_semantic_phrases(**context):
    """Task to extract semantic phrases from reviews"""
    try:
        libraries = setup_libraries()
        df = connect_and_fetch_data()
        df = preprocess_data(df)
        
        # Extract semantic phrases for positive reviews
        logging.info("Processing positive reviews...")
        df['Semantic Phrases Pos'] = extract_semantic_phrases_batch(
            df['positive_review_translated'].tolist(), libraries
        )
        
        # Extract semantic phrases for negative reviews
        logging.info("Processing negative reviews...")
        df['Semantic Phrases Neg'] = extract_semantic_phrases_batch(
            df['negative_review_translated'].tolist(), libraries
        )
        
        # Clear GPU cache if available
        if libraries['torch_available'] and libraries['device'] == 'cuda':
            libraries['torch'].cuda.empty_cache()
            logging.info("GPU cache cleared")
        
        # Store results
        context['task_instance'].xcom_push(key='semantic_phrases_extracted', value=True)
        context['task_instance'].xcom_push(key='records_with_phrases', value=len(df))
        
        logging.info(f"Semantic phrases extracted for {len(df)} records")
        return f"Extracted semantic phrases for {len(df)} records"
    except Exception as e:
        logging.error(f"Semantic phrase extraction failed: {e}")
        raise

def task_extract_tourism_tokens(**context):
    """Task to extract and filter tourism tokens"""
    try:
        libraries = setup_libraries()
        df = connect_and_fetch_data()
        df = preprocess_data(df)
        
        # Re-extract semantic phrases (needed for token extraction)
        df['Semantic Phrases Pos'] = extract_semantic_phrases_batch(
            df['positive_review_translated'].tolist(), libraries
        )
        df['Semantic Phrases Neg'] = extract_semantic_phrases_batch(
            df['negative_review_translated'].tolist(), libraries
        )
        
        # Extract and clean tokens
        logging.info("Extracting and cleaning tokens...")
        positive_tokens = extract_and_clean_tokens(df['Semantic Phrases Pos'], libraries)
        negative_tokens = extract_and_clean_tokens(df['Semantic Phrases Neg'], libraries)
        
        # Filter for tourism-relevant tokens
        tourism_positive_tokens = filter_tourism_tokens(positive_tokens, TOURISM_VOCABULARY)
        tourism_negative_tokens = filter_tourism_tokens(negative_tokens, TOURISM_VOCABULARY)
        
        # Apply tourism token extraction to dataframe
        df['Tourism_Tokens_Pos'] = df['Semantic Phrases Pos'].apply(
            lambda x: extract_tourism_tokens_per_row(x, tourism_positive_tokens)
        )
        df['Tourism_Tokens_Neg'] = df['Semantic Phrases Neg'].apply(
            lambda x: extract_tourism_tokens_per_row(x, tourism_negative_tokens)
        )
        
        # Store results
        context['task_instance'].xcom_push(key='positive_tokens_count', value=len(tourism_positive_tokens))
        context['task_instance'].xcom_push(key='negative_tokens_count', value=len(tourism_negative_tokens))
        context['task_instance'].xcom_push(key='records_with_tokens', value=len(df))
        
        logging.info(f"Tourism tokens extracted - Positive: {len(tourism_positive_tokens)}, Negative: {len(tourism_negative_tokens)}")
        return f"Extracted {len(tourism_positive_tokens)} positive and {len(tourism_negative_tokens)} negative tourism tokens"
    except Exception as e:
        logging.error(f"Tourism token extraction failed: {e}")
        raise

def task_prepare_final_nlp_dataset(**context):
    """Task to prepare final dataset for database ingestion"""
    try:
        libraries = setup_libraries()
        df = connect_and_fetch_data()
        df = preprocess_data(df)
        
        # Re-extract semantic phrases and tourism tokens
        df['Semantic Phrases Pos'] = extract_semantic_phrases_batch(
            df['positive_review_translated'].tolist(), libraries
        )
        df['Semantic Phrases Neg'] = extract_semantic_phrases_batch(
            df['negative_review_translated'].tolist(), libraries
        )
        
        positive_tokens = extract_and_clean_tokens(df['Semantic Phrases Pos'], libraries)
        negative_tokens = extract_and_clean_tokens(df['Semantic Phrases Neg'], libraries)
        tourism_positive_tokens = filter_tourism_tokens(positive_tokens, TOURISM_VOCABULARY)
        tourism_negative_tokens = filter_tourism_tokens(negative_tokens, TOURISM_VOCABULARY)
        
        df['Tourism_Tokens_Pos'] = df['Semantic Phrases Pos'].apply(
            lambda x: extract_tourism_tokens_per_row(x, tourism_positive_tokens)
        )
        df['Tourism_Tokens_Neg'] = df['Semantic Phrases Neg'].apply(
            lambda x: extract_tourism_tokens_per_row(x, tourism_negative_tokens)
        )
        
        # Filter out reviews with no tourism tokens
        df = df[~((df['Tourism_Tokens_Pos'].apply(len) == 0) &
                  (df['Tourism_Tokens_Neg'].apply(len) == 0))].copy()
        
        # Prepare final dataset
        final_df = prepare_final_dataset(df)
        
        # Store results
        context['task_instance'].xcom_push(key='final_dataset_count', value=len(final_df))
        context['task_instance'].xcom_push(key='filtered_reviews_count', value=len(df))
        
        logging.info(f"Final NLP dataset prepared with {len(final_df)} records")
        return f"Prepared final dataset with {len(final_df)} records"
    except Exception as e:
        logging.error(f"Final dataset preparation failed: {e}")
        raise

def task_ingest_nlp_data(**context):
    """Task to ingest processed NLP data to database"""
    try:
        libraries = setup_libraries()
        df = connect_and_fetch_data()
        df = preprocess_data(df)
        
        # Re-extract all processing steps
        df['Semantic Phrases Pos'] = extract_semantic_phrases_batch(
            df['positive_review_translated'].tolist(), libraries
        )
        df['Semantic Phrases Neg'] = extract_semantic_phrases_batch(
            df['negative_review_translated'].tolist(), libraries
        )
        
        positive_tokens = extract_and_clean_tokens(df['Semantic Phrases Pos'], libraries)
        negative_tokens = extract_and_clean_tokens(df['Semantic Phrases Neg'], libraries)
        tourism_positive_tokens = filter_tourism_tokens(positive_tokens, TOURISM_VOCABULARY)
        tourism_negative_tokens = filter_tourism_tokens(negative_tokens, TOURISM_VOCABULARY)
        
        df['Tourism_Tokens_Pos'] = df['Semantic Phrases Pos'].apply(
            lambda x: extract_tourism_tokens_per_row(x, tourism_positive_tokens)
        )
        df['Tourism_Tokens_Neg'] = df['Semantic Phrases Neg'].apply(
            lambda x: extract_tourism_tokens_per_row(x, tourism_negative_tokens)
        )
        
        df = df[~((df['Tourism_Tokens_Pos'].apply(len) == 0) &
                  (df['Tourism_Tokens_Neg'].apply(len) == 0))].copy()
        
        final_df = prepare_final_dataset(df)
        
        # Ingest to database
        ingest_to_database(final_df)
        
        # Store ingestion results
        context['task_instance'].xcom_push(key='ingested_records_count', value=len(final_df))
        
        logging.info(f"NLP data ingestion completed for {len(final_df)} records")
        return f"Ingested {len(final_df)} records to gold.final_reviews_test"
    except Exception as e:
        logging.error(f"NLP data ingestion failed: {e}")
        raise

def task_nlp_processing_report(**context):
    """Task to generate NLP processing report"""
    try:
        # Get results from previous tasks
        ti = context['task_instance']
        nlp_data_count = ti.xcom_pull(task_ids='fetch_nlp_data', key='nlp_data_count') or 0
        positive_tokens = ti.xcom_pull(task_ids='extract_tourism_tokens', key='positive_tokens_count') or 0
        negative_tokens = ti.xcom_pull(task_ids='extract_tourism_tokens', key='negative_tokens_count') or 0
        final_dataset_count = ti.xcom_pull(task_ids='prepare_final_nlp_dataset', key='final_dataset_count') or 0
        ingested_count = ti.xcom_pull(task_ids='ingest_nlp_data', key='ingested_records_count') or 0
        torch_available = ti.xcom_pull(task_ids='setup_nlp_libraries', key='torch_available') or False
        keybert_available = ti.xcom_pull(task_ids='setup_nlp_libraries', key='keybert_available') or False
        device = ti.xcom_pull(task_ids='setup_nlp_libraries', key='device') or 'unknown'
        
        report = f"""
        NLP Processing Pipeline Report
        =============================
        
        Execution Date: {context['ds']}
        
        Environment:
        - PyTorch available: {torch_available}
        - KeyBERT available: {keybert_available}
        - Processing device: {device}
        
        Data Processing:
        - Source records: {nlp_data_count:,}
        - Tourism tokens (positive): {positive_tokens:,}
        - Tourism tokens (negative): {negative_tokens:,}
        - Final dataset records: {final_dataset_count:,}
        - Records ingested to gold layer: {ingested_count:,}
        
        Status: {'SUCCESS' if ingested_count > 0 else 'FAILED'}
        """
        
        logging.info(report)
        
        # Store report in XCom
        context['task_instance'].xcom_push(key='nlp_processing_report', value=report)
        
        return report
        
    except Exception as e:
        logging.error(f"NLP processing report generation failed: {e}")
        raise

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

# Create TaskGroup for scraping related tasks
with TaskGroup(group_id='Scraping_hotels_reviews', dag=dag) as scraping_group:
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

    truncate_reviews_table_task = PythonOperator(
        task_id='truncate_reviews_table',
        python_callable=task_truncate_reviews_table,
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

    # Define internal task group dependencies
    get_links_task >> setup_driver_task >> truncate_reviews_table_task >> scrape_task >> report_task

cleanup_task = PythonOperator(
    task_id='cleanup',
    python_callable=task_cleanup,
    dag=dag,
    trigger_rule='all_done'  # Run even if previous tasks failed
)

# Create TaskGroup for preprocessing and translation tasks
with TaskGroup(group_id='Preprocessing_and_Translating_non-eng', dag=dag) as preprocessing_translation_group:
    preprocessing_setup_task = PythonOperator(
        task_id='preprocessing_setup',
        python_callable=task_preprocessing_setup,
        dag=dag
    )

    preprocessing_task = PythonOperator(
        task_id='data_preprocessing',
        python_callable=task_data_preprocessing,
        dag=dag,
        execution_timeout=timedelta(hours=2),
        retries=1,
        retry_delay=timedelta(minutes=10)
    )

    preprocessing_validation_task = PythonOperator(
        task_id='preprocessing_validation',
        python_callable=task_preprocessing_validation,
        dag=dag
    )

    translation_setup_task = PythonOperator(
        task_id='translation_setup',
        python_callable=task_translation_setup,
        dag=dag
    )

    load_data_for_translation_task = PythonOperator(
        task_id='load_data_for_translation',
        python_callable=task_load_data_for_translation,
        dag=dag
    )

    translate_reviews_task = PythonOperator(
        task_id='translate_reviews',
        python_callable=task_translate_reviews,
        dag=dag,
        execution_timeout=timedelta(hours=3),
        retries=1,
        retry_delay=timedelta(minutes=15),
        max_active_tis_per_dag=1
    )

    ingest_translated_data_task = PythonOperator(
        task_id='ingest_translated_data',
        python_callable=task_ingest_translated_data,
        dag=dag,
        execution_timeout=timedelta(hours=1),
        retries=1,
        retry_delay=timedelta(minutes=10)
    )

    translation_validation_task = PythonOperator(
        task_id='translation_validation',
        python_callable=task_translation_validation,
        dag=dag
    )

    translation_report_task = PythonOperator(
        task_id='translation_report',
        python_callable=task_translation_report,
        dag=dag
    )

    # Define internal task group dependencies
    preprocessing_setup_task >> preprocessing_task >> preprocessing_validation_task >> translation_setup_task >> load_data_for_translation_task >> translate_reviews_task >> ingest_translated_data_task >> translation_validation_task >> translation_report_task

# Create TaskGroup for NLP processing tasks
with TaskGroup(group_id='NLP_Processing_Pipeline', dag=dag) as nlp_processing_group:
    setup_nlp_libraries_task = PythonOperator(
        task_id='setup_nlp_libraries',
        python_callable=task_setup_nlp_libraries,
        dag=dag
    )

    fetch_nlp_data_task = PythonOperator(
        task_id='fetch_nlp_data',
        python_callable=task_fetch_nlp_data,
        dag=dag
    )

    preprocess_nlp_data_task = PythonOperator(
        task_id='preprocess_nlp_data',
        python_callable=task_preprocess_nlp_data,
        dag=dag
    )

    extract_semantic_phrases_task = PythonOperator(
        task_id='extract_semantic_phrases',
        python_callable=task_extract_semantic_phrases,
        dag=dag,
        execution_timeout=timedelta(hours=2),
        retries=1,
        retry_delay=timedelta(minutes=15)
    )

    extract_tourism_tokens_task = PythonOperator(
        task_id='extract_tourism_tokens',
        python_callable=task_extract_tourism_tokens,
        dag=dag,
        execution_timeout=timedelta(hours=1),
        retries=1,
        retry_delay=timedelta(minutes=10)
    )

    prepare_final_nlp_dataset_task = PythonOperator(
        task_id='prepare_final_nlp_dataset',
        python_callable=task_prepare_final_nlp_dataset,
        dag=dag
    )

    ingest_nlp_data_task = PythonOperator(
        task_id='ingest_nlp_data',
        python_callable=task_ingest_nlp_data,
        dag=dag,
        execution_timeout=timedelta(hours=1),
        retries=1,
        retry_delay=timedelta(minutes=10)
    )

    nlp_processing_report_task = PythonOperator(
        task_id='nlp_processing_report',
        python_callable=task_nlp_processing_report,
        dag=dag
    )

    # Define internal task group dependencies
    setup_nlp_libraries_task >> fetch_nlp_data_task >> preprocess_nlp_data_task >> extract_semantic_phrases_task >> extract_tourism_tokens_task >> prepare_final_nlp_dataset_task >> ingest_nlp_data_task >> nlp_processing_report_task

# Define task dependencies
init_logging_task >> setup_db_task >> scraping_group >> cleanup_task >> preprocessing_translation_group >> nlp_processing_group

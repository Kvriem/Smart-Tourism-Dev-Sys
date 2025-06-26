import re
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import functools
from sqlalchemy import create_engine, text
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import logging
import time
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('preprocessing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Download NLTK resources
logger.info("Downloading NLTK resources...")
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
logger.info("NLTK resources downloaded successfully")

# Database connection string
DB_CONNECTION_STRING = "postgresql://neondb_owner:npg_ExFXHY8yiNT0@ep-lingering-term-ab7pbfql-pooler.eu-west-2.aws.neon.tech/neondb?sslmode=require"

# Compile regex patterns once for better performance
logger.info("Compiling regex patterns...")
NON_ALPHA_PATTERN = re.compile(r'[^a-zA-Z\s]')
MULTI_SPACE_PATTERN = re.compile(r'\s{2,}')
DURATION_PATTERN = re.compile(r'[^0-9a-zA-Z\s]')
logger.info("Regex patterns compiled successfully")

# Cache stopwords set
logger.info("Loading stopwords...")
STOPWORDS_SET = set(stopwords.words('english'))
logger.info(f"Loaded {len(STOPWORDS_SET)} stopwords")

# ======================
# DATABASE FUNCTIONS
# ======================
def create_db_engine():
    """Create database engine"""
    logger.info("Creating database engine...")
    try:
        engine = create_engine(DB_CONNECTION_STRING)
        logger.info("Database engine created successfully")
        return engine
    except Exception as e:
        logger.error(f"Failed to create database engine: {str(e)}")
        raise

def ensure_schema_and_table_exist(engine):
    """Ensure silver schema and reviews_cleaned table exist"""
    logger.info("Ensuring silver schema and table exist...")
    try:
        with engine.connect() as conn:
            # Create silver schema if it doesn't exist
            logger.info("Creating silver schema if not exists...")
            conn.execute(text("CREATE SCHEMA IF NOT EXISTS silver"))
            logger.info("Silver schema ensured")
            
            # Drop existing table if it exists to ensure correct schema
            logger.info("Dropping existing reviews_cleaned table if exists...")
            conn.execute(text("DROP TABLE IF EXISTS silver.reviews_cleaned"))
            logger.info("Existing table dropped")
            
            # Create reviews_cleaned table with correct schema
            logger.info("Creating reviews_cleaned table...")
            create_table_sql = """
            CREATE TABLE silver.reviews_cleaned (
                id SERIAL PRIMARY KEY,
                city TEXT,
                hotel_name TEXT,
                reviewer_name TEXT,
                reviewer_nationality TEXT,
                duration TEXT,
                check_in_date TEXT,
                travel_type TEXT,
                room_type TEXT,
                review_date TEXT,
                positive_review TEXT,
                negative_review TEXT,
                ingestion_timestamp TIMESTAMP,
                sentiment_classification INTEGER
            )
            """
            conn.execute(text(create_table_sql))
            logger.info("reviews_cleaned table created successfully")
    except Exception as e:
        logger.error(f"Failed to ensure schema and table: {str(e)}")
        raise

def read_from_bronze_table(engine):
    """Read data from bronze table using direct psycopg2 connection"""
    query = """
    SELECT * FROM bronze.hotels_reviews_test
    """
    
    try:
        # Use psycopg2 connection directly with pandas
        import psycopg2
        from urllib.parse import urlparse
        
        # Parse the connection string
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
        logger.info(f"Successfully read {len(df)} records from bronze.hotels_reviews_test")
        return df
    except Exception as e:
        logger.error(f"Error reading from database: {e}")
        raise

def ingest_to_silver_table(df, engine):
    """Ingest processed data to silver.reviews_cleaned table"""
    logger.info(f"Starting ingestion of {len(df)} records to silver.reviews_cleaned...")
    try:
        start_time = time.time()
        
        # Define the columns we want to keep in the silver table (matching bronze schema + sentiment)
        silver_columns = [
            "city", "hotel_name", "reviewer_name", "reviewer_nationality", 
            "duration", "check_in_date", "travel_type", "room_type", 
            "review_date", "positive_review", "negative_review", 
            "ingestion_timestamp", "sentiment_classification"        ]
        
        # Filter DataFrame to only include columns that exist and we want
        existing_columns = [col for col in silver_columns if col in df.columns]
        df_filtered = df[existing_columns].copy()
        
        logger.info(f"Filtered DataFrame to {len(existing_columns)} columns: {existing_columns}")
        # Truncate existing data
        logger.info("Truncating existing data from silver.reviews_cleaned...")
        with engine.connect() as conn:
            # Use a transaction to handle the truncate operation
            with conn.begin():
                conn.execute(text("TRUNCATE TABLE silver.reviews_cleaned"))
        logger.info("Existing data truncated")        # Insert new data using direct SQL insert statements
        logger.info("Inserting processed data...")
        
        # Manually insert data using SQLAlchemy
        with engine.connect() as conn:
            # Use a transaction for the inserts
            with conn.begin():
                # Create the insert statement using SQLAlchemy parameter syntax
                insert_sql = """
                INSERT INTO silver.reviews_cleaned 
                (city, hotel_name, reviewer_name, reviewer_nationality, duration, 
                 check_in_date, travel_type, room_type, review_date, positive_review, 
                 negative_review, ingestion_timestamp, sentiment_classification)
                VALUES (:city, :hotel_name, :reviewer_name, :reviewer_nationality, :duration, 
                        :check_in_date, :travel_type, :room_type, :review_date, :positive_review, 
                        :negative_review, :ingestion_timestamp, :sentiment_classification)
                """
                
                # Convert DataFrame to records for insertion
                records = df_filtered.to_dict('records')
                
                # Insert the records
                for record in records:
                    conn.execute(text(insert_sql), record)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Successfully ingested {len(df_filtered)} records to silver table in {elapsed_time:.2f} seconds")
    except Exception as e:
        logger.error(f"Failed to ingest data to silver table: {str(e)}")
        raise

# ======================
# TEXT PROCESSING FUNCTIONS
# ======================
def remove_stopwords_vectorized(series):
    """Vectorized stopword removal"""
    logger.debug(f"Removing stopwords from {len(series)} texts...")
    start_time = time.time()
    
    def remove_sw(text):
        if not isinstance(text, str) or not text.strip():
            return text
        tokens = word_tokenize(text)
        return ' '.join([w for w in tokens if w.lower() not in STOPWORDS_SET])
    
    result = series.apply(remove_sw)
    elapsed_time = time.time() - start_time
    logger.debug(f"Stopword removal completed in {elapsed_time:.2f} seconds")
    return result

def classify_sentiment(row):
    """Classify review sentiment based on content"""
    sentiment_list = ['nothing', 'negative feedback', 'positive review', 0]
    neg_review = row['negative_review']
    pos_review = row['positive_review']
    
    if (neg_review not in sentiment_list and 
        pos_review not in sentiment_list):
        return -1  # Both reviews present
    elif neg_review not in sentiment_list:
        return 0   # Only negative review
    elif pos_review not in sentiment_list:
        return 1   # Only positive review
    return 2        # No substantive reviews

# ======================
# MAIN PROCESSING FUNCTIONS
# ======================
def process_chunk(df):
    """Process a chunk of data with preprocessing only (no translation)"""
    logger.info(f"Processing chunk with {len(df)} records...")
    start_time = time.time()
    
    # Ensure we're working with a proper copy
    df = df.copy()
    logger.debug("Created DataFrame copy")
    
    # Log the columns we're working with
    logger.info(f"Input columns: {list(df.columns)}")
    
    # Remove duplicates
    original_count = len(df)
    df = df.drop_duplicates()
    logger.info(f"Removed {original_count - len(df)} duplicates, {len(df)} records remaining")

    # Add ingestion timestamp
    df.loc[:, 'ingestion_timestamp'] = pd.Timestamp.now()
    logger.info("Added ingestion timestamp")

    # Extract city (vectorized with better performance)
    logger.info("Starting city extraction...")
    cities = ['Alexandria', 'Sharm El Sheikh', 'Luxor', 'Aswan', 'Hurghada', 'Cairo']
    city_pattern = '|'.join([re.escape(city.lower()) for city in cities])
    city_regex = re.compile(city_pattern)
    
    def fast_extract_city_vectorized(x):
        if not isinstance(x, str):
            return None
        match = city_regex.search(x.lower())
        if match:
            for city in cities:
                if city.lower() in x.lower():
                    return city
        return None
    
    if 'city' in df.columns:
        city_start_time = time.time()
        df.loc[:, 'city'] = df['city'].map(fast_extract_city_vectorized)
        city_extracted_count = df['city'].notna().sum()
        logger.info(f"City extraction completed in {time.time() - city_start_time:.2f} seconds. Extracted {city_extracted_count} cities")

    # Enhanced vectorized text cleaning
    logger.info("Starting text cleaning process...")
    text_clean_start_time = time.time()
    
    def clean_text_optimized(series):
        series = series.astype(str)
        series = series.str.replace(NON_ALPHA_PATTERN, ' ', regex=True)
        series = series.str.replace(MULTI_SPACE_PATTERN, ' ', regex=True)
        series = series.str.strip().str.lower()
        return series

    text_columns = [col for col in ['hotel_name', 'negative_review', 'positive_review'] if col in df.columns]
    logger.info(f"Cleaning text for columns: {text_columns}")
    
    for col in text_columns:
        logger.debug(f"Cleaning column: {col}")
        df.loc[:, col] = clean_text_optimized(df[col])
        df.loc[:, col] = remove_stopwords_vectorized(df[col])
    
    logger.info(f"Text cleaning completed in {time.time() - text_clean_start_time:.2f} seconds")

    # Optimized duration cleaning
    if 'duration' in df.columns:
        logger.info("Cleaning duration column...")
        df.loc[:, 'duration'] = df['duration'].astype(str).str.replace(DURATION_PATTERN, ' ', regex=True)
        logger.info("Duration cleaning completed")

    # Optimized review date cleaning
    if 'review_date' in df.columns:
        logger.info("Cleaning review date column...")
        df.loc[:, 'review_date'] = df['review_date'].astype(str).str.replace('Reviewed: ', '', regex=False)
        logger.info("Review date cleaning completed")

    # Vectorized empty review handling
    logger.info("Handling empty reviews...")
    if 'negative_review' in df.columns:
        empty_neg_before = (df['negative_review'] == '').sum()
        df.loc[:, 'negative_review'] = df['negative_review'].replace('', 0)
        logger.info(f"Replaced {empty_neg_before} empty negative reviews with 0")
        
    if 'positive_review' in df.columns:
        empty_pos_before = (df['positive_review'] == '').sum()
        df.loc[:, 'positive_review'] = df['positive_review'].replace('', 0)
        logger.info(f"Replaced {empty_pos_before} empty positive reviews with 0")

    # Sentiment classification
    if 'negative_review' in df.columns and 'positive_review' in df.columns:
        logger.info("Starting sentiment classification...")
        sentiment_start_time = time.time()
        df.loc[:, 'sentiment_classification'] = df.apply(classify_sentiment, axis=1)
        
        sentiment_counts = df['sentiment_classification'].value_counts().sort_index()
        logger.info(f"Sentiment classification completed in {time.time() - sentiment_start_time:.2f} seconds")
        logger.info(f"Sentiment distribution: {sentiment_counts.to_dict()}")

    elapsed_time = time.time() - start_time
    logger.info(f"Chunk processing completed in {elapsed_time:.2f} seconds")
    logger.info(f"Output columns: {list(df.columns)}")
    return df

def process_reviews_data_from_db(engine):
    """Process reviews data from database"""
    logger.info("Starting reviews data preprocessing from database...")
    overall_start_time = time.time()
    
    df = read_from_bronze_table(engine)
    logger.info(f"Loaded {len(df)} records from bronze.hotels_reviews")

    # Process in chunks if dataset is large
    if len(df) > 50000:
        logger.info(f"Large dataset detected ({len(df)} records). Processing in chunks...")
        chunk_size = 10000
        processed_chunks = []
        total_chunks = (len(df) + chunk_size - 1) // chunk_size
        
        for i in range(0, len(df), chunk_size):
            chunk_num = i // chunk_size + 1
            logger.info(f"Processing chunk {chunk_num}/{total_chunks} (rows {i} to {min(i+chunk_size, len(df))})")
            
            chunk = df.iloc[i:i+chunk_size].copy()
            processed_chunk = process_chunk(chunk)
            processed_chunks.append(processed_chunk)
            logger.info(f"Chunk {chunk_num}/{total_chunks} completed")
        
        logger.info("Concatenating processed chunks...")
        concat_start_time = time.time()
        df = pd.concat(processed_chunks, ignore_index=True)
        logger.info(f"Chunks concatenated in {time.time() - concat_start_time:.2f} seconds")
    else:
        logger.info(f"Small dataset ({len(df)} records). Processing as single chunk...")
        df = df.copy()
        df = process_chunk(df)

    overall_elapsed_time = time.time() - overall_start_time
    logger.info(f"Preprocessing complete! Processed data shape: {df.shape}")
    logger.info(f"Total processing time: {overall_elapsed_time:.2f} seconds")
    logger.info(f"Average processing rate: {len(df)/overall_elapsed_time:.2f} records/second")
    
    return df

# ======================
# EXECUTION
# ======================
if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("STARTING PREPROCESSING (SILVER LAYER)")
    logger.info("=" * 60)
    
    execution_start_time = time.time()
    
    try:
        # Create database engine
        logger.info("Step 1: Creating database connection...")
        engine = create_db_engine()
        logger.info("✓ Database connection established")
        
        # Ensure schema and table exist
        logger.info("Step 2: Setting up database schema and tables...")
        ensure_schema_and_table_exist(engine)
        logger.info("✓ Silver schema and reviews_cleaned table ensured")
        
        # Process reviews data from database
        logger.info("Step 3: Preprocessing reviews data...")
        reviews_df = process_reviews_data_from_db(engine)
        logger.info(f"✓ Preprocessing complete! Processed data shape: {reviews_df.shape}")
        
        # Ingest processed data to silver table
        logger.info("Step 4: Ingesting data to silver table...")
        ingest_to_silver_table(reviews_df, engine)
        logger.info("✓ Data successfully ingested to silver.reviews_cleaned")
        
        total_execution_time = time.time() - execution_start_time
        logger.info("=" * 60)
        logger.info("PREPROCESSING COMPLETED SUCCESSFULLY")
        logger.info(f"Total execution time: {total_execution_time:.2f} seconds")
        logger.info(f"Final dataset size: {reviews_df.shape}")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error("=" * 60)
        logger.error("PREPROCESSING FAILED")
        logger.error(f"Error occurred: {str(e)}")
        logger.error("=" * 60)
        raise
    finally:
        if 'engine' in locals():
            logger.info("Closing database connection...")
            engine.dispose()
            logger.info("✓ Database connection closed")


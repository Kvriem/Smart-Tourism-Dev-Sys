#!/usr/bin/env python3
"""
Tourism Review Processing Pipeline - Functional Script Version
Restructured from final_optmized__notebook.ipynb
Maintains exact same output structure and process order
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import psycopg2
from sqlalchemy import create_engine, text
import json
import re
import nltk
from collections import Counter
import time
import logging
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/home/kariem/airflow/logs/tourism_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def install_dependencies():
    """Install required packages"""
    import subprocess
    import sys
    
    packages = [
        "keybert", "sentence-transformers", "torch", "tqdm", 
        "nltk", "psycopg2-binary", "sqlalchemy", "pandas"
    ]
    
    for package in packages:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])


def setup_database_and_gpu():
    """Setup database connection and check GPU availability"""
    # Database connection
    DATABASE_URL = "postgresql://neondb_owner:npg_ExFXHY8yiNT0@ep-lingering-term-ab7pbfql-pooler.eu-west-2.aws.neon.tech/neondb?sslmode=require"
    
    # Create SQLAlchemy engine
    engine = create_engine(DATABASE_URL)
    
    # Check GPU availability
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        device = 'cuda'
    else:
        logger.info("Using CPU")
        device = 'cpu'
    
    return engine, device


def fetch_data_from_database(engine):
    """Fetch data from PostgreSQL database"""
    logger.info("Fetching data from silver.silver_translated...")
    try:
        df = pd.read_sql_query("SELECT * FROM silver.silver_translated", engine)
        logger.info(f"Dataset loaded from database: {df.shape}")
        logger.info("Database connection successful!")
        return df
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        return None


def display_basic_info(df, device):
    """Display basic dataset information"""
    logger.info("Dataset loaded successfully")
    logger.info(f"Columns: {list(df.columns)}")
    logger.info(f"Processing device: {device}")
    logger.info(f"Total reviews from database: {len(df)}")


def preview_data(df):
    """Quick data preview"""
    logger.info("Sample data from silver.silver_translated:")
    return df.head()


def check_missing_values(df):
    """Check for missing values in review columns"""
    missing_pos = df['Positive Review Translated'].isna().sum()
    missing_neg = df['Negative Review Translated'].isna().sum()
    logger.info(f"Missing positive reviews: {missing_pos}")
    logger.info(f"Missing negative reviews: {missing_neg}")
    logger.info("Data quality check completed")


def preprocess_data(df):
    """Data preprocessing - fill NaN values once"""
    df['Positive Review Translated'] = df['Positive Review Translated'].fillna('')
    df['Negative Review Translated'] = df['Negative Review Translated'].fillna('')
    logger.info("Data preprocessing completed")
    return df


def load_models(device):
    """Load models once - optimized for GPU"""
    logger.info("Loading models...")
    start_time = time.time()
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    kw_model = KeyBERT(model=embedding_model)
    logger.info(f"Models loaded in {time.time() - start_time:.2f} seconds")
    return kw_model


def extract_semantic_phrases_batch(texts, kw_model, batch_size=100):
    """Extract keywords in batches optimized for GPU"""
    all_results = []
    
    # Filter out empty texts upfront
    valid_indices = []
    valid_texts = []
    for i, text in enumerate(texts):
        if isinstance(text, str) and len(text.strip()) > 10:
            valid_indices.append(i)
            valid_texts.append(text)
    
    logger.info(f"Processing {len(valid_texts)} valid texts out of {len(texts)} total")
    
    # Process valid texts in batches
    valid_results = []
    for i in tqdm(range(0, len(valid_texts), batch_size), desc="Processing batches"):
        batch = valid_texts[i:i+batch_size]
        batch_results = []
        
        for text in batch:
            try:
                keywords = kw_model.extract_keywords(
                    text,
                    keyphrase_ngram_range=(1, 2),
                    stop_words='english',
                    use_maxsum=False,
                    nr_candidates=10,
                    top_n=3
                )
                batch_results.append([kw[0] for kw in keywords])
            except Exception as e:
                batch_results.append([])
        
        valid_results.extend(batch_results)
    
    # Map results back to original indices
    result_map = dict(zip(valid_indices, valid_results))
    all_results = [result_map.get(i, []) for i in range(len(texts))]
    
    return all_results


def process_semantic_extraction(df, kw_model, device):
    """Process positive and negative reviews for semantic extraction"""
    # Process positive reviews
    logger.info("Processing positive reviews...")
    start_time = time.time()
    positive_texts = df['Positive Review Translated'].tolist()
    df['Semantic Phrases Pos'] = extract_semantic_phrases_batch(positive_texts, kw_model)
    pos_time = time.time() - start_time
    logger.info(f"Positive reviews processed in {pos_time:.2f} seconds")

    # Process negative reviews
    logger.info("Processing negative reviews...")
    start_time = time.time()
    negative_texts = df['Negative Review Translated'].tolist()
    df['Semantic Phrases Neg'] = extract_semantic_phrases_batch(negative_texts, kw_model)
    neg_time = time.time() - start_time
    logger.info(f"Negative reviews processed in {neg_time:.2f} seconds")

    # Summary statistics
    missing_pos = sum(1 for x in df['Semantic Phrases Pos'] if len(x) == 0)
    missing_neg = sum(1 for x in df['Semantic Phrases Neg'] if len(x) == 0)
    logger.info(f"Empty semantic phrases in positive reviews: {missing_pos}")
    logger.info(f"Empty semantic phrases in negative reviews: {missing_neg}")
    logger.info(f"Total processing time: {pos_time + neg_time:.2f} seconds")

    # GPU memory cleanup
    if device == 'cuda':
        torch.cuda.empty_cache()
        logger.info("GPU cache cleared")
    
    return df


def verify_semantic_results(df):
    """Verify semantic extraction results"""
    logger.info(f"Dataset shape after semantic extraction: {df.shape}")
    logger.info(f"Semantic phrase columns: {[col for col in df.columns if 'Semantic Phrases' in col]}")


def preview_semantic_phrases(df):
    """Sample semantic phrases preview"""
    sample_results = df[['Positive Review Translated', 'Semantic Phrases Pos', 
                        'Negative Review Translated', 'Semantic Phrases Neg']].head(10)
    for idx, row in sample_results.iterrows():
        if len(row['Semantic Phrases Pos']) > 0:
            logger.info(f"Row {idx}:")
            logger.info(f"Positive phrases: {row['Semantic Phrases Pos']}")
            if len(row['Semantic Phrases Neg']) > 0:
                logger.info(f"Negative phrases: {row['Semantic Phrases Neg']}")
            logger.info("-" * 50)


def setup_nltk_and_vocabulary():
    """Setup NLTK data and define vocabulary"""
    # Download required NLTK data
    try:
        nltk.download('stopwords', quiet=True)
        nltk.download('names', quiet=True)
        from nltk.corpus import stopwords, names
        stop_words = set(stopwords.words('english'))
        name_list = set(names.words())
    except:
        logger.warning("NLTK data not available, using basic filtering")
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        name_list = set()

    # Additional hotel-specific stop words to filter
    hotel_stopwords = {
        'hotel', 'room', 'stay', 'night', 'day', 'time', 'place', 'location', 'area', 'staff', 'service', 
        'people', 'guest', 'customer', 'visitor', 'person', 'man', 'woman', 'guy', 'lady', 'someone', 
        'anyone', 'everyone', 'everything', 'something', 'anything', 'nothing'
    }

    # Comprehensive tourism services and interests vocabulary
    tourism_vocabulary = {
        # Accommodation & Room Features
        'accommodation', 'suite', 'apartment', 'villa', 'cottage', 'cabin', 'chalet', 'hostel', 'guesthouse', 'resort',
        'bedroom', 'bathroom', 'kitchen', 'balcony', 'terrace', 'patio', 'garden', 'pool', 'jacuzzi', 'spa',
        'wifi', 'internet', 'television', 'minibar', 'refrigerator', 'airconditioning', 'heating', 'fireplace',
        'bed', 'pillow', 'mattress', 'linen', 'towel', 'amenities', 'toiletries', 'hairdryer', 'safe', 'wardrobe',
        
        # Hotel Services & Facilities
        'reception', 'concierge', 'housekeeping', 'maintenance', 'security', 'valet', 'bellhop', 'porter',
        'checkin', 'checkout', 'reservation', 'booking', 'availability', 'upgrade', 'complimentary', 'inclusive',
        'restaurant', 'dining', 'breakfast', 'lunch', 'dinner', 'buffet', 'menu', 'cuisine', 'bar', 'lounge',
        'gym', 'fitness', 'sauna', 'massage', 'wellness', 'relaxation', 'treatment', 'therapy',
        'conference', 'meeting', 'business', 'events', 'wedding', 'banquet', 'catering',
        'parking', 'garage', 'transportation', 'shuttle', 'airport', 'transfer', 'taxi', 'rental',
        
        # Tourism Activities & Attractions
        'sightseeing', 'tour', 'excursion', 'adventure', 'exploration', 'hiking', 'walking', 'cycling', 'biking',
        'museum', 'gallery', 'exhibition', 'theater', 'cinema', 'entertainment', 'nightlife', 'shopping',
        'beach', 'ocean', 'sea', 'lake', 'river', 'mountain', 'forest', 'park', 'nature', 'landscape',
        'historic', 'cultural', 'heritage', 'architecture', 'monument', 'castle', 'church', 'temple',
        'festival', 'event', 'celebration', 'carnival', 'market', 'fair', 'concert', 'performance',
        
        # Experience & Service Quality
        'comfortable', 'luxury', 'elegant', 'modern', 'traditional', 'authentic', 'unique', 'spectacular',
        'clean', 'spacious', 'cozy', 'quiet', 'peaceful', 'relaxing', 'convenient', 'accessible',
        'friendly', 'helpful', 'professional', 'courteous', 'attentive', 'responsive', 'efficient',
        'delicious', 'tasty', 'fresh', 'quality', 'variety', 'selection', 'choice', 'option',
        'expensive', 'affordable', 'reasonable', 'value', 'price', 'cost', 'budget', 'cheap',
        'recommend', 'satisfaction', 'experience', 'memorable', 'enjoyable', 'pleasant', 'disappointing',
        
        # Location & Geography
        'downtown', 'center', 'district', 'neighborhood', 'vicinity', 'nearby', 'walking', 'distance',
        'view', 'scenery', 'panoramic', 'overlooking', 'facing', 'waterfront', 'beachfront', 'hillside',
        'accessibility', 'transportation', 'connection', 'proximity', 'convenience', 'central',
        
        # Negative Aspects (Important for feedback)
        'problem', 'issue', 'complaint', 'dissatisfied', 'unhappy', 'poor', 'worst', 'terrible',
        'dirty', 'noisy', 'crowded', 'outdated', 'broken', 'damaged', 'faulty',
        'rude', 'unfriendly', 'unprofessional', 'slow', 'delayed', 'cancelled', 'overbooked',
        'overpriced', 'overrated', 'disappointing', 'unacceptable', 'uncomfortable'
    }
    
    return stop_words, hotel_stopwords, name_list, tourism_vocabulary


def extract_and_clean_tokens(phrase_lists, stop_words, hotel_stopwords, name_list):
    """Extract individual tokens from phrase lists and remove redundant/non-meaningful ones"""
    all_tokens = []
    
    for phrase_list in phrase_lists:
        if isinstance(phrase_list, list):
            for phrase in phrase_list:
                if isinstance(phrase, str):
                    # Split phrase into individual words
                    tokens = re.findall(r'\b[a-z]+\b', phrase.lower())
                    all_tokens.extend(tokens)
    
    # Remove duplicates and filter tokens
    unique_tokens = set(all_tokens)
    
    # Filter out unwanted tokens
    filtered_tokens = []
    for token in unique_tokens:
        if (len(token) >= 3 and
            token not in stop_words and
            token not in hotel_stopwords and
            token.lower() not in name_list and
            not token.isdigit() and
            token.isalpha()):
            filtered_tokens.append(token)
    
    return sorted(filtered_tokens)


def process_token_extraction(df, stop_words, hotel_stopwords, name_list):
    """Extract and clean tokens"""
    logger.info("Extracting and cleaning tokens...")
    positive_tokens = extract_and_clean_tokens(df['Semantic Phrases Pos'], stop_words, hotel_stopwords, name_list)
    negative_tokens = extract_and_clean_tokens(df['Semantic Phrases Neg'], stop_words, hotel_stopwords, name_list)
    logger.info(f"Positive tokens extracted: {len(positive_tokens)}")
    logger.info(f"Negative tokens extracted: {len(negative_tokens)}")
    return positive_tokens, negative_tokens


def filter_tourism_tokens(token_list, tourism_vocab):
    """Filter tokens to only include tourism-related terms"""
    tourism_tokens = []
    for token in token_list:
        if token.lower() in tourism_vocab:
            tourism_tokens.append(token)
    return sorted(tourism_tokens)


def categorize_tourism_tokens(tokens):
    """Categorize tokens by tourism service types"""
    categories = {
        'accommodation': [],
        'services': [],
        'activities': [],
        'dining': [],
        'quality': [],
        'location': [],
        'issues': []
    }
    
    # Define category keywords
    accommodation_terms = {'accommodation', 'suite', 'apartment', 'villa', 'bedroom', 'bathroom', 'kitchen', 'balcony', 'pool', 'spa', 'bed', 'amenities'}
    service_terms = {'reception', 'concierge', 'housekeeping', 'checkin', 'checkout', 'reservation', 'parking', 'transportation', 'shuttle'}
    activity_terms = {'sightseeing', 'tour', 'excursion', 'museum', 'beach', 'hiking', 'shopping', 'entertainment', 'nightlife'}
    dining_terms = {'restaurant', 'dining', 'breakfast', 'lunch', 'dinner', 'buffet', 'cuisine', 'bar', 'cafe', 'wine', 'delicious'}
    quality_terms = {'comfortable', 'luxury', 'clean', 'spacious', 'friendly', 'professional', 'quality', 'recommend', 'satisfaction'}
    location_terms = {'downtown', 'center', 'nearby', 'view', 'scenery', 'accessibility', 'convenient', 'central'}
    issue_terms = {'problem', 'complaint', 'dirty', 'noisy', 'broken', 'rude', 'slow', 'overpriced', 'disappointing'}
    
    for token in tokens:
        token_lower = token.lower()
        if token_lower in accommodation_terms:
            categories['accommodation'].append(token)
        elif token_lower in service_terms:
            categories['services'].append(token)
        elif token_lower in activity_terms:
            categories['activities'].append(token)
        elif token_lower in dining_terms:
            categories['dining'].append(token)
        elif token_lower in quality_terms:
            categories['quality'].append(token)
        elif token_lower in location_terms:
            categories['location'].append(token)
        elif token_lower in issue_terms:
            categories['issues'].append(token)
    
    return categories


def filter_and_categorize_tourism_tokens(positive_tokens, negative_tokens, tourism_vocabulary):
    """Filter for tourism-relevant tokens only and categorize them"""
    logger.info("Filtering tokens for tourism relevance...")
    tourism_positive_tokens = filter_tourism_tokens(positive_tokens, tourism_vocabulary)
    tourism_negative_tokens = filter_tourism_tokens(negative_tokens, tourism_vocabulary)

    logger.info(f"Tourism-relevant positive tokens: {len(tourism_positive_tokens)} out of {len(positive_tokens)}")
    logger.info(f"Tourism-relevant negative tokens: {len(tourism_negative_tokens)} out of {len(negative_tokens)}")

    # Categorize filtered tokens
    pos_categories = categorize_tourism_tokens(tourism_positive_tokens)
    neg_categories = categorize_tourism_tokens(tourism_negative_tokens)

    logger.info("=== POSITIVE TOKEN CATEGORIES ===")
    for category, tokens in pos_categories.items():
        if tokens:
            logger.info(f"{category.upper()}: {len(tokens)} tokens - {tokens[:10]}")

    logger.info("=== NEGATIVE TOKEN CATEGORIES ===")
    for category, tokens in neg_categories.items():
        if tokens:
            logger.info(f"{category.upper()}: {len(tokens)} tokens - {tokens[:10]}")
    
    return tourism_positive_tokens, tourism_negative_tokens


def extract_tourism_tokens_per_row(phrase_list, valid_tourism_tokens):
    """Extract only tourism-relevant tokens for each row"""
    if not isinstance(phrase_list, list):
        return []
    
    row_tokens = set()
    for phrase in phrase_list:
        if isinstance(phrase, str):
            tokens = re.findall(r'\b[a-z]+\b', phrase.lower())
            for token in tokens:
                if token in [t.lower() for t in valid_tourism_tokens]:
                    row_tokens.add(token)
    
    return sorted(list(row_tokens))


def apply_tourism_filtering(df, tourism_positive_tokens, tourism_negative_tokens):
    """Apply tourism filtering to dataframe rows"""
    logger.info("Creating final tourism-filtered token columns...")
    df['Tourism_Tokens_Pos'] = df['Semantic Phrases Pos'].apply(
        lambda x: extract_tourism_tokens_per_row(x, tourism_positive_tokens)
    )
    df['Tourism_Tokens_Neg'] = df['Semantic Phrases Neg'].apply(
        lambda x: extract_tourism_tokens_per_row(x, tourism_negative_tokens)
    )

    # Final statistics
    tourism_pos_count = sum(len(tokens) for tokens in df['Tourism_Tokens_Pos'])
    tourism_neg_count = sum(len(tokens) for tokens in df['Tourism_Tokens_Neg'])

    logger.info("=== FINAL TOURISM TOKEN RESULTS ===")
    logger.info(f"Tourism-filtered positive tokens: {tourism_pos_count}")
    logger.info(f"Tourism-filtered negative tokens: {tourism_neg_count}")
    logger.info(f"Average positive tourism tokens per review: {tourism_pos_count / len(df):.2f}")
    logger.info(f"Average negative tourism tokens per review: {tourism_neg_count / len(df):.2f}")
    
    return df


def filter_empty_tourism_reviews(df):
    """Filter out reviews with empty tourism token lists"""
    logger.info("=== FILTERING EMPTY TOURISM TOKEN REVIEWS ===")

    # Check current dataset statistics
    logger.info(f"Dataset before filtering: {len(df)} reviews")

    # Count reviews with empty tourism tokens
    empty_pos_tokens = (df['Tourism_Tokens_Pos'].apply(len) == 0).sum()
    empty_neg_tokens = (df['Tourism_Tokens_Neg'].apply(len) == 0).sum()
    both_empty = ((df['Tourism_Tokens_Pos'].apply(len) == 0) & 
                  (df['Tourism_Tokens_Neg'].apply(len) == 0)).sum()

    logger.info(f"Reviews with empty positive tourism tokens: {empty_pos_tokens}")
    logger.info(f"Reviews with empty negative tourism tokens: {empty_neg_tokens}")
    logger.info(f"Reviews with both empty tourism token lists: {both_empty}")

    # Filter out reviews where both positive and negative tourism tokens are empty
    df_filtered = df[~((df['Tourism_Tokens_Pos'].apply(len) == 0) & 
                       (df['Tourism_Tokens_Neg'].apply(len) == 0))].copy()

    logger.info(f"Dataset after filtering: {len(df_filtered)} reviews")
    logger.info(f"Removed {len(df) - len(df_filtered)} reviews with no tourism tokens")
    logger.info(f"Retention rate: {(len(df_filtered) / len(df)) * 100:.1f}%")

    # Update statistics for filtered dataset
    filtered_pos_count = sum(len(tokens) for tokens in df_filtered['Tourism_Tokens_Pos'])
    filtered_neg_count = sum(len(tokens) for tokens in df_filtered['Tourism_Tokens_Neg'])

    logger.info("=== FILTERED DATASET STATISTICS ===")
    logger.info(f"Total positive tourism tokens: {filtered_pos_count}")
    logger.info(f"Total negative tourism tokens: {filtered_neg_count}")
    logger.info(f"Average positive tokens per review: {filtered_pos_count / len(df_filtered):.2f}")
    logger.info(f"Average negative tokens per review: {filtered_neg_count / len(df_filtered):.2f}")

    # Show distribution of token counts
    pos_token_counts = df_filtered['Tourism_Tokens_Pos'].apply(len)
    neg_token_counts = df_filtered['Tourism_Tokens_Neg'].apply(len)

    logger.info("Positive token distribution:")
    logger.info(f"  Min: {pos_token_counts.min()}, Max: {pos_token_counts.max()}")
    logger.info(f"  Mean: {pos_token_counts.mean():.2f}, Median: {pos_token_counts.median():.1f}")

    logger.info("Negative token distribution:")
    logger.info(f"  Min: {neg_token_counts.min()}, Max: {neg_token_counts.max()}")
    logger.info(f"  Mean: {neg_token_counts.mean():.2f}, Median: {neg_token_counts.median():.1f}")

    logger.info(f"Dataset updated to filtered version with {len(df_filtered)} informative reviews")
    
    return df_filtered


def analyze_sentiment_patterns(df):
    """Analyze sentiment-specific patterns"""
    if 'sentiment classification' in df.columns:
        negative_sentiment = df[df['sentiment classification'] == -1]['Tourism_Tokens_Neg']
        non_empty_neg = [tokens for tokens in negative_sentiment if len(tokens) > 0]
        logger.info(f"Negative sentiment reviews with tourism tokens: {len(non_empty_neg)}")
        if len(non_empty_neg) > 0:
            logger.info(f"Sample negative tourism tokens: {non_empty_neg[:5]}")


def display_sample_tourism_results(df):
    """Display sample tourism-filtered results"""
    logger.info("=== SAMPLE TOURISM-FILTERED RESULTS ===")
    sample_rows = df[['Tourism_Tokens_Pos', 'Tourism_Tokens_Neg']].head(15)
    for idx, row in sample_rows.iterrows():
        if len(row['Tourism_Tokens_Pos']) > 0 or len(row['Tourism_Tokens_Neg']) > 0:
            logger.info(f"Row {idx}:")
            if len(row['Tourism_Tokens_Pos']) > 0:
                logger.info(f"  Positive Tourism Tokens: {row['Tourism_Tokens_Pos']}")
            if len(row['Tourism_Tokens_Neg']) > 0:
                logger.info(f"  Negative Tourism Tokens: {row['Tourism_Tokens_Neg']}")
            logger.info("")


def convert_to_postgres_array(token_list):
    """Convert Python list to PostgreSQL TEXT[] format"""
    if not isinstance(token_list, list) or len(token_list) == 0:
        return None
    # Escape any quotes in the tokens and format as PostgreSQL array
    escaped_tokens = []
    for token in token_list:
        escaped_token = token.replace('"', '""')
        escaped_tokens.append(f'"{escaped_token}"')
    return '{' + ','.join(escaped_tokens) + '}'


def prepare_target_schema_data(df):
    """Prepare data for database ingestion with target schema"""
    logger.info("=== PREPARING DATA FOR TARGET SCHEMA ===")

    # Create final dataframe with only required columns and proper naming
    final_df = pd.DataFrame()

    # Map source columns to target schema
    final_df['city'] = df['City']
    final_df['hotel_name'] = df['Hotel Name']
    final_df['reviewer_name'] = df['Reviewer Name']
    final_df['reviewer_nationality'] = df['Reviewer Nationality']
    final_df['duration'] = df['Duration']
    final_df['check_in_date'] = df['Check-in Date']
    final_df['review_date'] = df['Review Date']
    final_df['travel_type'] = df['Travel Type']
    final_df['room_type'] = df['Room Type']
    final_df['positive_review'] = df['Positive Review Translated']
    final_df['negative_review'] = df['Negative Review Translated']
    final_df['sentiment_classification'] = df['sentiment classification']

    # Convert tourism tokens to PostgreSQL TEXT[] format
    final_df['positive_tokens'] = df['Tourism_Tokens_Pos'].apply(convert_to_postgres_array)
    final_df['negative_tokens'] = df['Tourism_Tokens_Neg'].apply(convert_to_postgres_array)

    logger.info(f"Final dataset prepared for target schema: {final_df.shape}")
    logger.info(f"Target columns: {list(final_df.columns)}")

    # Verify data types and content
    logger.info("Data type verification:")
    for col in final_df.columns:
        logger.info(f"  {col}: {final_df[col].dtype}")

    logger.info("Sample token arrays:")
    non_null_pos = final_df['positive_tokens'].dropna()
    non_null_neg = final_df['negative_tokens'].dropna()
    if len(non_null_pos) > 0:
        logger.info(f"  Positive tokens sample: {non_null_pos.iloc[0]}")
    if len(non_null_neg) > 0:
        logger.info(f"  Negative tokens sample: {non_null_neg.iloc[0]}")
    
    return final_df


def ingest_to_database(engine, final_df):
    """Create target table schema and ingest data using append mode"""
    logger.info("=== DATABASE INGESTION WITH APPEND MODE ===")

    try:
        with engine.connect() as conn:
            # Create gold schema if it doesn't exist
            conn.execute(text("CREATE SCHEMA IF NOT EXISTS gold"))
            logger.info("✅ Gold schema created/verified")
            
            # Check if table exists, if not create it
            table_exists_query = """
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_schema = 'gold' AND table_name = 'final_reviews'
            )
            """
            table_exists = conn.execute(text(table_exists_query)).scalar()
            
            if not table_exists:
                # Create table with exact target schema only if it doesn't exist
                create_table_sql = """
                CREATE TABLE gold.final_reviews (
                    id SERIAL PRIMARY KEY,
                    city TEXT,
                    hotel_name TEXT,
                    reviewer_name TEXT,
                    reviewer_nationality TEXT,
                    duration TEXT,
                    check_in_date TEXT,
                    review_date TEXT,
                    travel_type TEXT,
                    room_type TEXT,
                    positive_review TEXT,
                    negative_review TEXT,
                    sentiment_classification INTEGER,
                    positive_tokens TEXT[],
                    negative_tokens TEXT[],
                    inserted_at TIMESTAMP DEFAULT now()
                )
                """
                
                conn.execute(text(create_table_sql))
                logger.info("✅ Target table created with proper schema")
            else:
                logger.info("✅ Target table already exists, will append data")
            
            # Get current record count before insertion
            count_before = conn.execute(text("SELECT COUNT(*) FROM gold.final_reviews")).scalar()
            logger.info(f"Records in table before insertion: {count_before}")
            
            # Commit schema changes
            conn.commit()
        
        # Ingest data using manual INSERT for proper TEXT[] handling
        logger.info("Appending processed data to gold.final_reviews...")
        
        with engine.connect() as conn:
            # Prepare INSERT statement
            insert_sql = """
            INSERT INTO gold.final_reviews (
                city, hotel_name, reviewer_name, reviewer_nationality, duration,
                check_in_date, review_date, travel_type, room_type, positive_review,
                negative_review, sentiment_classification, positive_tokens, negative_tokens
            ) VALUES (
                :city, :hotel_name, :reviewer_name, :reviewer_nationality, :duration,
                :check_in_date, :review_date, :travel_type, :room_type, :positive_review,
                :negative_review, :sentiment_classification, :positive_tokens, :negative_tokens
            )
            """
            
            # Insert data in batches
            batch_size = 1000
            total_rows = len(final_df)
            inserted_count = 0
            
            for i in tqdm(range(0, total_rows, batch_size), desc="Inserting batches"):
                batch = final_df.iloc[i:i+batch_size]
                batch_data = []
                
                for _, row in batch.iterrows():
                    batch_data.append({
                        'city': row['city'],
                        'hotel_name': row['hotel_name'],
                        'reviewer_name': row['reviewer_name'],
                        'reviewer_nationality': row['reviewer_nationality'],
                        'duration': row['duration'],
                        'check_in_date': row['check_in_date'],
                        'review_date': row['review_date'],
                        'travel_type': row['travel_type'],
                        'room_type': row['room_type'],
                        'positive_review': row['positive_review'],
                        'negative_review': row['negative_review'],
                        'sentiment_classification': row['sentiment_classification'],
                        'positive_tokens': row['positive_tokens'],
                        'negative_tokens': row['negative_tokens']
                    })
                
                result = conn.execute(text(insert_sql), batch_data)
                inserted_count += len(batch_data)
                conn.commit()
                
                # Log progress every 5 batches
                if (i // batch_size + 1) % 5 == 0:
                    logger.info(f"Inserted {inserted_count} records so far...")
        
        # Verify ingestion
        with engine.connect() as conn:
            count_after = conn.execute(text("SELECT COUNT(*) FROM gold.final_reviews")).scalar()
            newly_inserted = count_after - count_before
            
            logger.info("✅ Data ingestion completed!")
            logger.info(f"✅ Records inserted in this run: {newly_inserted}")
            logger.info(f"✅ Total records in table: {count_after}")
            logger.info(f"✅ Expected insertions: {len(final_df)}")
            
            if newly_inserted == len(final_df):
                logger.info("✅ All records successfully appended")
            else:
                logger.warning(f"⚠️  Mismatch: Expected {len(final_df)}, but inserted {newly_inserted}")
            
            # Verify schema matches target (only log if table was just created)
            if not table_exists:
                schema_result = conn.execute(text("""
                    SELECT column_name, data_type, is_nullable 
                    FROM information_schema.columns 
                    WHERE table_schema = 'gold' AND table_name = 'final_reviews'
                    ORDER BY ordinal_position
                """))
                
                logger.info("✅ Target schema verification:")
                for col_name, data_type, nullable in schema_result.fetchall():
                    logger.info(f"   {col_name}: {data_type} ({'NULL' if nullable == 'YES' else 'NOT NULL'})")
                
    except Exception as e:
        logger.error(f"❌ Database ingestion failed: {e}")
        logger.warning("Saving to local CSV as backup...")
        backup_filename = f'backup_final_reviews_{int(time.time())}.csv'
        final_df.to_csv(backup_filename, index=False)
        logger.info(f"✅ Backup saved to {backup_filename}")


def print_project_summary(final_df):
    """Final project summary with append mode"""
    logger.info("=== TOURISM PROCESSING WITH APPEND MODE COMPLETE ===")
    logger.info("✅ Data fetched from silver.silver_translated")
    logger.info("✅ Semantic phrase extraction and tourism token filtering applied")
    logger.info("✅ Data mapped to target schema with proper column names")
    logger.info("✅ Tourism tokens converted to PostgreSQL TEXT[] format")
    logger.info("✅ Target table created/verified with exact schema specification")
    logger.info("✅ Data appended to gold.final_reviews with proper data types")

    logger.info("Target Schema Summary:")
    logger.info("- Source: silver.silver_translated")
    logger.info("- Destination: gold.final_reviews (APPEND MODE)")
    logger.info("- Schema: 17 columns (id, 13 data columns, positive_tokens[], negative_tokens[], inserted_at)")
    logger.info(f"- Records processed in this run: {len(final_df)}")
    logger.info("- Token arrays: PostgreSQL TEXT[] format")
    logger.info("- Mode: Incremental load (append to existing data)")

    logger.info("Column Mapping:")
    logger.info("- City → city (TEXT)")
    logger.info("- Hotel Name → hotel_name (TEXT)")
    logger.info("- Positive Review Translated → positive_review (TEXT)")
    logger.info("- Negative Review Translated → negative_review (TEXT)")
    logger.info("- Tourism_Tokens_Pos → positive_tokens (TEXT[])")
    logger.info("- Tourism_Tokens_Neg → negative_tokens (TEXT[])")
    logger.info("- sentiment classification → sentiment_classification (INTEGER)")


def verify_target_schema_compliance(engine):
    """Final verification of target schema compliance with append mode details"""
    logger.info("=== TARGET SCHEMA COMPLIANCE VERIFICATION ===")

    try:
        with engine.connect() as conn:
            # Get table statistics
            stats_query = """
            SELECT 
                COUNT(*) as total_records,
                COUNT(DISTINCT inserted_at::date) as distinct_load_dates,
                MIN(inserted_at) as first_load,
                MAX(inserted_at) as latest_load
            FROM gold.final_reviews
            """
            
            stats_result = conn.execute(text(stats_query)).fetchone()
            if stats_result:
                logger.info("✅ Table statistics:")
                logger.info(f"   Total records: {stats_result[0]}")
                logger.info(f"   Distinct load dates: {stats_result[1]}")
                logger.info(f"   First load: {stats_result[2]}")
                logger.info(f"   Latest load: {stats_result[3]}")

            # Verify exact schema match
            verification_query = """
            SELECT 
                column_name,
                data_type,
                CASE WHEN column_default LIKE 'nextval%' THEN 'SERIAL' ELSE data_type END as display_type,
                is_nullable,
                column_default
            FROM information_schema.columns 
            WHERE table_schema = 'gold' AND table_name = 'final_reviews'
            ORDER BY ordinal_position
            """
            
            result = conn.execute(text(verification_query))
            columns = result.fetchall()
            
            logger.info("✅ Final table schema verification:")
            for col_name, data_type, display_type, nullable, default in columns:
                default_info = f" DEFAULT {default}" if default else ""
                logger.info(f"   {col_name}: {display_type.upper()}{default_info}")
            
            # Verify token array functionality
            token_test = conn.execute(text("""
                SELECT 
                    COUNT(*) as total_records,
                    COUNT(positive_tokens) as records_with_pos_tokens,
                    COUNT(negative_tokens) as records_with_neg_tokens,
                    array_length(positive_tokens, 1) as sample_pos_length,
                    array_length(negative_tokens, 1) as sample_neg_length
                FROM gold.final_reviews 
                WHERE positive_tokens IS NOT NULL OR negative_tokens IS NOT NULL
                LIMIT 1
            """))
            
            token_stats = token_test.fetchone()
            if token_stats:
                logger.info("✅ Token array verification:")
                logger.info(f"   Total records: {token_stats[0]}")
                logger.info(f"   Records with positive tokens: {token_stats[1]}")
                logger.info(f"   Records with negative tokens: {token_stats[2]}")
                if token_stats[3]:
                    logger.info(f"   Sample positive token array length: {token_stats[3]}")
                if token_stats[4]:
                    logger.info(f"   Sample negative token array length: {token_stats[4]}")
            
            # Test array querying capability
            array_query_test = conn.execute(text("""
                SELECT positive_tokens[1] as first_positive_token
                FROM gold.final_reviews 
                WHERE positive_tokens IS NOT NULL 
                LIMIT 1
            """))
            
            first_token = array_query_test.scalar()
            if first_token:
                logger.info(f"   Array indexing test successful: '{first_token}'")
            
    except Exception as e:
        logger.error(f"❌ Schema verification failed: {e}")


def main():
    """Main execution function"""
    logger.info("=== TOURISM REVIEW PROCESSING PIPELINE ===")
    
    # Step 1: Setup
    engine, device = setup_database_and_gpu()
    
    # Step 2: Fetch data
    df = fetch_data_from_database(engine)
    if df is None:
        return
    
    # Step 3: Display basic info
    display_basic_info(df, device)
    preview_data(df)
    check_missing_values(df)
    
    # Step 4: Preprocess data
    df = preprocess_data(df)
    
    # Step 5: Load models and process semantic extraction
    kw_model = load_models(device)
    df = process_semantic_extraction(df, kw_model, device)
    
    # Step 6: Verify results
    verify_semantic_results(df)
    preview_semantic_phrases(df)
    
    # Step 7: Setup vocabulary and extract tokens
    stop_words, hotel_stopwords, name_list, tourism_vocabulary = setup_nltk_and_vocabulary()
    positive_tokens, negative_tokens = process_token_extraction(df, stop_words, hotel_stopwords, name_list)
    
    # Step 8: Filter and categorize tourism tokens
    tourism_positive_tokens, tourism_negative_tokens = filter_and_categorize_tourism_tokens(
        positive_tokens, negative_tokens, tourism_vocabulary
    )
    
    # Step 9: Apply tourism filtering
    df = apply_tourism_filtering(df, tourism_positive_tokens, tourism_negative_tokens)
    
    # Step 10: Filter empty reviews
    df = filter_empty_tourism_reviews(df)
    
    # Step 11: Analyze patterns and display samples
    analyze_sentiment_patterns(df)
    display_sample_tourism_results(df)
    
    # Step 12: Prepare target schema and ingest
    final_df = prepare_target_schema_data(df)
    ingest_to_database(engine, final_df)
    
    # Step 13: Final summary and verification
    print_project_summary(final_df)
    verify_target_schema_compliance(engine)
    
    # Step 14: Cleanup
    engine.dispose()
    logger.info("✅ Database connections closed")
    logger.info("✅ ETL Pipeline with target schema completed successfully!")
    logger.info("✅ Ready for tourism analytics with PostgreSQL TEXT[] token arrays")


if __name__ == "__main__":
    main()

"""
NLP Processing Pipeline for Hotel Reviews
Extracts semantic phrases and tourism-relevant tokens from hotel reviews
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
import psycopg2
from psycopg2.extras import RealDictCursor
import json
import time
import re
import nltk
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer

# Database configuration
DB_CONFIG = {
    'host': 'ep-lingering-term-ab7pbfql-pooler.eu-west-2.aws.neon.tech',
    'database': 'neondb',
    'user': 'neondb_owner',
    'password': 'npg_ExFXHY8yiNT0',
    'port': 5432,
    'sslmode': 'require'
}

# Tourism vocabulary and stop words
HOTEL_STOPWORDS = {
    'hotel', 'room', 'stay', 'night', 'day', 'time', 'place', 'location', 'area', 'staff', 'service',
    'people', 'guest', 'customer', 'visitor', 'person', 'man', 'woman', 'guy', 'lady', 'someone',
    'anyone', 'everyone', 'everything', 'something', 'anything', 'nothing'
}

TOURISM_VOCABULARY = {
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
    # Negative Aspects
    'problem', 'issue', 'complaint', 'dissatisfied', 'unhappy', 'poor', 'worst', 'terrible',
    'dirty', 'noisy', 'crowded', 'outdated', 'broken', 'damaged', 'faulty',
    'rude', 'unfriendly', 'unprofessional', 'slow', 'delayed', 'cancelled', 'overbooked',
    'overpriced', 'overrated', 'disappointing', 'unacceptable', 'uncomfortable'
}


def setup_libraries():
    """Setup and import required libraries with fallbacks"""
    libraries = {}
    
    # Try to import torch
    try:
        import torch
        libraries['torch'] = torch
        libraries['torch_available'] = True
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            libraries['device'] = 'cuda'
        else:
            print("Using CPU with PyTorch")
            libraries['device'] = 'cpu'
    except ImportError:
        libraries['torch_available'] = False
        libraries['device'] = 'cpu'
        print("PyTorch not available - using basic text processing")
    
    # Try to import advanced NLP libraries
    try:
        from keybert import KeyBERT
        from sentence_transformers import SentenceTransformer
        libraries['keybert'] = KeyBERT
        libraries['sentence_transformers'] = SentenceTransformer
        libraries['keybert_available'] = True
        print("Advanced NLP libraries loaded successfully")
    except ImportError:
        libraries['keybert_available'] = False
        print("Advanced NLP libraries not available - using basic keyword extraction")
    
    # Setup NLTK
    try:
        nltk.download('stopwords', quiet=True)
        nltk.download('names', quiet=True)
        from nltk.corpus import stopwords, names
        libraries['stop_words'] = set(stopwords.words('english'))
        libraries['name_list'] = set(names.words())
    except:
        print("NLTK data not available, using basic filtering")
        libraries['stop_words'] = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        libraries['name_list'] = set()
    
    return libraries


def connect_and_fetch_data():
    """Connect to database and fetch data"""
    print("Connecting to PostgreSQL database...")
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        print("Fetching data from silver.silver_translated...")
        cur.execute("SELECT * FROM silver.silver_translated")
        rows = cur.fetchall()
        
        df = pd.DataFrame([dict(row) for row in rows])
        print(f"Dataset loaded from database: {df.shape}")
        print("Database connection successful!")
        
        cur.close()
        conn.close()
        
        return df
        
    except Exception as e:
        print(f"Database connection failed: {e}")
        return pd.DataFrame()


def preprocess_data(df):
    """Preprocess data by filling NaN values"""
    if not df.empty:
        df['positive_review_translated'] = df['positive_review_translated'].fillna('')
        df['negative_review_translated'] = df['negative_review_translated'].fillna('')
        print(f"Data preprocessing completed. Dataset shape: {df.shape}")
        print(f"Available columns: {list(df.columns)}")
    else:
        print("No data loaded - skipping preprocessing")
    return df


def extract_semantic_phrases_advanced(texts, libraries, batch_size=100):
    """Extract keywords using KeyBERT"""
    print("Loading KeyBERT models...")
    start_time = time.time()
    embedding_model = libraries['sentence_transformers']('all-MiniLM-L6-v2', device=libraries['device'])
    kw_model = libraries['keybert'](model=embedding_model)
    print(f"Models loaded in {time.time() - start_time:.2f} seconds")
    
    all_results = []
    valid_indices = []
    valid_texts = []
    
    for i, text in enumerate(texts):
        if isinstance(text, str) and len(text.strip()) > 10:
            valid_indices.append(i)
            valid_texts.append(text)

    print(f"Processing {len(valid_texts)} valid texts out of {len(texts)} total")

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
            except Exception:
                batch_results.append([])

        valid_results.extend(batch_results)

    result_map = dict(zip(valid_indices, valid_results))
    all_results = [result_map.get(i, []) for i in range(len(texts))]
    return all_results


def extract_semantic_phrases_fallback(texts, batch_size=100):
    """Extract keywords using TF-IDF as fallback"""
    all_results = []
    valid_texts = []
    valid_indices = []
    
    for i, text in enumerate(texts):
        if isinstance(text, str) and len(text.strip()) > 10:
            valid_indices.append(i)
            valid_texts.append(text)
    
    if not valid_texts:
        return [[] for _ in texts]
    
    vectorizer = TfidfVectorizer(
        max_features=1000,
        ngram_range=(1, 2),
        stop_words='english',
        min_df=1,
        max_df=0.8
    )
    
    try:
        tfidf_matrix = vectorizer.fit_transform(valid_texts)
        feature_names = vectorizer.get_feature_names_out()
        
        valid_results = []
        for i in range(len(valid_texts)):
            doc_scores = tfidf_matrix[i].toarray()[0]
            top_indices = doc_scores.argsort()[-3:][::-1]
            top_terms = [feature_names[idx] for idx in top_indices if doc_scores[idx] > 0]
            valid_results.append(top_terms)
    except:
        valid_results = [[] for _ in valid_texts]
    
    result_map = dict(zip(valid_indices, valid_results))
    all_results = [result_map.get(i, []) for i in range(len(texts))]
    return all_results


def extract_semantic_phrases_batch(texts, libraries, batch_size=100):
    """Extract semantic phrases using best available method"""
    if libraries['keybert_available'] and libraries['torch_available']:
        return extract_semantic_phrases_advanced(texts, libraries, batch_size)
    else:
        print("Using TF-IDF based keyword extraction...")
        return extract_semantic_phrases_fallback(texts, batch_size)


def extract_and_clean_tokens(phrase_lists, libraries):
    """Extract individual tokens from phrase lists and remove redundant ones"""
    all_tokens = []
    
    for phrase_list in phrase_lists:
        if isinstance(phrase_list, list):
            for phrase in phrase_list:
                if isinstance(phrase, str):
                    tokens = re.findall(r'\b[a-z]+\b', phrase.lower())
                    all_tokens.extend(tokens)
    
    unique_tokens = set(all_tokens)
    
    filtered_tokens = []
    for token in unique_tokens:
        if (len(token) >= 3 and
            token not in libraries['stop_words'] and
            token not in HOTEL_STOPWORDS and
            token.lower() not in libraries['name_list'] and
            not token.isdigit() and
            token.isalpha()):
            filtered_tokens.append(token)
    
    return sorted(filtered_tokens)


def filter_tourism_tokens(token_list, tourism_vocab):
    """Filter tokens to only include tourism-related terms"""
    return sorted([token for token in token_list if token.lower() in tourism_vocab])


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


def convert_to_postgres_array(token_list):
    """Convert Python list to PostgreSQL TEXT[] format"""
    if not isinstance(token_list, list) or len(token_list) == 0:
        return None
    escaped_tokens = []
    for token in token_list:
        escaped_token = str(token).replace("'", "''")
        escaped_tokens.append(f"'{escaped_token}'")
    return '{' + ','.join(escaped_tokens) + '}'


def prepare_final_dataset(df):
    """Prepare data for database ingestion"""
    if df.empty:
        print("No data to prepare for ingestion")
        return pd.DataFrame()
    
    final_df = pd.DataFrame()
    
    # Map source columns to target schema
    final_df['city'] = df['city']
    final_df['hotel_name'] = df['hotel_name']
    final_df['reviewer_name'] = df['reviewer_name']
    final_df['reviewer_nationality'] = df['reviewer_nationality']
    final_df['duration'] = df['duration']
    final_df['check_in_date'] = df['check_in_date']
    final_df['review_date'] = df['review_date']
    final_df['travel_type'] = df['travel_type']
    final_df['room_type'] = df['room_type']
    final_df['positive_review'] = df['positive_review_translated']
    final_df['negative_review'] = df['negative_review_translated']
    final_df['sentiment_classification'] = df['sentiment_classification']
    
    # Convert tourism tokens to PostgreSQL format
    final_df['positive_tokens'] = df['Tourism_Tokens_Pos'].apply(convert_to_postgres_array)
    final_df['negative_tokens'] = df['Tourism_Tokens_Neg'].apply(convert_to_postgres_array)
    
    print(f"Final dataset prepared: {final_df.shape}")
    return final_df


def ingest_to_database(final_df):
    """Ingest data to database with duplicate prevention"""
    print("=== DATABASE INGESTION (APPEND MODE) ===")
    
    if final_df.empty:
        print("No data to ingest")
        return
    
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        
        # Create schema and table
        cur.execute("CREATE SCHEMA IF NOT EXISTS gold")
        
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS gold.final_reviews_test (
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
        
        cur.execute(create_table_sql)
        conn.commit()
        print("✅ Target table ready")
        
        # Check existing records
        cur.execute("SELECT COUNT(*) FROM gold.final_reviews_test")
        existing_count = cur.fetchone()[0]
        print(f"📊 Existing records in table: {existing_count}")
        
        # Insert data with duplicate prevention
        insert_sql = """
        INSERT INTO gold.final_reviews_test (
            city, hotel_name, reviewer_name, reviewer_nationality, duration,
            check_in_date, review_date, travel_type, room_type, positive_review,
            negative_review, sentiment_classification, positive_tokens, negative_tokens
        )
        SELECT %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
        WHERE NOT EXISTS (
            SELECT 1 FROM gold.final_reviews_test f
            WHERE f.city = %s 
            AND f.hotel_name = %s
            AND f.reviewer_name = %s
            AND f.review_date = %s
        )
        """
        
        batch_size = 1000
        inserted_count = 0
        
        for i in tqdm(range(0, len(final_df), batch_size), desc="Inserting data"):
            batch = final_df.iloc[i:i+batch_size]
            
            for _, row in batch.iterrows():
                insert_data = (
                    row['city'], row['hotel_name'], row['reviewer_name'], 
                    row['reviewer_nationality'], row['duration'], row['check_in_date'],
                    row['review_date'], row['travel_type'], row['room_type'],
                    row['positive_review'], row['negative_review'], 
                    row['sentiment_classification'], row['positive_tokens'], 
                    row['negative_tokens'],
                    # Duplicate check parameters
                    row['city'], row['hotel_name'], row['reviewer_name'], row['review_date']
                )
                
                cur.execute(insert_sql, insert_data)
                if cur.rowcount > 0:
                    inserted_count += 1
            
            conn.commit()
        
        print(f"✅ Records actually inserted: {inserted_count}")
        
        # Verify final count
        cur.execute("SELECT COUNT(*) FROM gold.final_reviews_test")
        final_count = cur.fetchone()[0]
        print(f"📊 Final table count: {final_count}")
        print(f"📈 Net new records added: {final_count - existing_count}")
        
        cur.close()
        conn.close()
        
    except Exception as e:
        print(f"❌ Database ingestion failed: {e}")
        final_df.to_csv('backup_final_reviews.csv', index=False)
        print("✅ Backup saved to CSV")
        
        if 'cur' in locals():
            cur.close()
        if 'conn' in locals():
            conn.close()


def main():
    """Main execution function"""
    print("Starting NLP Processing Pipeline...")
    
    # Setup libraries
    libraries = setup_libraries()
    
    # Load and preprocess data
    df = connect_and_fetch_data()
    if df.empty:
        print("No data available. Exiting.")
        return
    
    df = preprocess_data(df)
    
    # Extract semantic phrases
    print("Processing positive reviews...")
    df['Semantic Phrases Pos'] = extract_semantic_phrases_batch(
        df['positive_review_translated'].tolist(), libraries
    )
    
    print("Processing negative reviews...")
    df['Semantic Phrases Neg'] = extract_semantic_phrases_batch(
        df['negative_review_translated'].tolist(), libraries
    )
    
    # Clear GPU cache if available
    if libraries['torch_available'] and libraries['device'] == 'cuda':
        libraries['torch'].cuda.empty_cache()
        print("GPU cache cleared")
    
    # Extract and filter tokens
    print("Extracting and cleaning tokens...")
    positive_tokens = extract_and_clean_tokens(df['Semantic Phrases Pos'], libraries)
    negative_tokens = extract_and_clean_tokens(df['Semantic Phrases Neg'], libraries)
    print(f"Positive tokens extracted: {len(positive_tokens)}")
    print(f"Negative tokens extracted: {len(negative_tokens)}")
    
    # Filter for tourism-relevant tokens
    tourism_positive_tokens = filter_tourism_tokens(positive_tokens, TOURISM_VOCABULARY)
    tourism_negative_tokens = filter_tourism_tokens(negative_tokens, TOURISM_VOCABULARY)
    
    print(f"Tourism-relevant positive tokens: {len(tourism_positive_tokens)}")
    print(f"Tourism-relevant negative tokens: {len(tourism_negative_tokens)}")
    
    # Apply tourism token extraction to dataframe
    df['Tourism_Tokens_Pos'] = df['Semantic Phrases Pos'].apply(
        lambda x: extract_tourism_tokens_per_row(x, tourism_positive_tokens)
    )
    df['Tourism_Tokens_Neg'] = df['Semantic Phrases Neg'].apply(
        lambda x: extract_tourism_tokens_per_row(x, tourism_negative_tokens)
    )
    
    # Filter out reviews with no tourism tokens
    df = df[~((df['Tourism_Tokens_Pos'].apply(len) == 0) &
              (df['Tourism_Tokens_Neg'].apply(len) == 0))].copy()
    
    print(f"Final dataset with tourism tokens: {len(df)} reviews")
    
    # Prepare and ingest data
    final_df = prepare_final_dataset(df)
    ingest_to_database(final_df)
    
    print("✅ ETL Pipeline completed!")


if __name__ == "__main__":
    main()

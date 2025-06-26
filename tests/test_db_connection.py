import pandas as pd
import sqlalchemy
from sqlalchemy import create_engine, text
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_database_connection():
    """Test database connection and bronze_reviews table access"""
    
    # Neon database connection string
    connection_string = "postgresql://neondb_owner:npg_ExFXHY8yiNT0@ep-lingering-term-ab7pbfql-pooler.eu-west-2.aws.neon.tech/neondb?sslmode=require"
    
    try:
        # Test 1: Create engine
        logger.info("Creating SQLAlchemy engine...")
        engine = create_engine(connection_string)
        logger.info(f"Engine created: {type(engine)}")
        
        # Test 2: Test basic connection
        logger.info("Testing basic connection...")
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            logger.info("✓ Basic connection successful")
        
        # Test 3: Check if bronze_reviews table exists
        logger.info("Checking if bronze_reviews table exists...")
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_name = 'hotels_reviews_test'
            """))
            tables = result.fetchall()
            if tables:
                logger.info("✓ hotels_reviews_test table exists")
            else:
                logger.error("✗ hotels_reviews_test table not found")
                return

        # Test 4: Count rows in hotels_reviews_test
        logger.info("Counting rows in hotels_reviews_test...")
        with engine.connect() as conn:
            result = conn.execute(text("SELECT COUNT(*) FROM bronze_reviews"))
            count = result.fetchone()[0]
            logger.info(f"✓ bronze_reviews has {count} rows")
        
        # Test 5: Test pandas read_sql with connection
        logger.info("Testing pandas read_sql with connection...")
        query = "SELECT review_id, review_text, stars, date FROM bronze.hotels_reviews_test LIMIT 5"
        with engine.connect() as conn:
            df = pd.read_sql(query, conn)
            logger.info(f"✓ Successfully read {len(df)} rows with pandas")
            logger.info(f"Columns: {list(df.columns)}")
        
        # Test 6: Test pandas read_sql with engine directly
        logger.info("Testing pandas read_sql with engine directly...")
        try:
            df = pd.read_sql(query, engine)
            logger.info(f"✓ Successfully read {len(df)} rows with engine")
        except Exception as e:
            logger.error(f"✗ Failed to read with engine: {e}")
        
        logger.info("All tests completed!")
        
    except Exception as e:
        logger.error(f"Connection test failed: {e}")
        return False
    
    return True



if __name__ == "__main__":
    test_database_connection()

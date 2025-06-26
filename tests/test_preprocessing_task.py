#!/usr/bin/env python3
"""
Test script for preprocessing task
"""
import sys
import os

# Add the src directory to Python path
sys.path.append('/home/kariem/airflow/src')

def test_preprocessing_imports():
    """Test if all preprocessing imports work"""
    print("Testing preprocessing imports...")
    try:
        from nlp.preprocessing_nlp_ref.preprocessing_script import (
            create_db_engine,
            ensure_schema_and_table_exist,
            read_from_bronze_table,
            process_reviews_data_from_db,
            ingest_to_silver_table
        )
        print("✅ All preprocessing imports successful!")
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_preprocessing_setup():
    """Test preprocessing setup"""
    print("\nTesting preprocessing setup...")
    try:
        from nlp.preprocessing_nlp_ref.preprocessing_script import create_db_engine
        
        # Test database engine creation
        engine = create_db_engine()
        print("✅ Database engine created successfully!")
        
        # Test connection
        with engine.connect() as conn:
            result = conn.execute("SELECT 1")
            print("✅ Database connection test successful!")
        
        return engine
    except Exception as e:
        print(f"❌ Setup error: {e}")
        return None

def test_schema_and_table():
    """Test schema and table creation"""
    print("\nTesting schema and table creation...")
    try:
        from nlp.preprocessing_nlp_ref.preprocessing_script import (
            create_db_engine,
            ensure_schema_and_table_exist
        )
        
        engine = create_db_engine()
        ensure_schema_and_table_exist(engine)
        print("✅ Schema and table creation successful!")
        return True
    except Exception as e:
        print(f"❌ Schema/table error: {e}")
        return False

def test_read_bronze_data():
    """Test reading from bronze table"""
    print("\nTesting bronze data reading...")
    try:
        from nlp.preprocessing_nlp_ref.preprocessing_script import (
            create_db_engine,
            read_from_bronze_table
        )
        
        engine = create_db_engine()
        df = read_from_bronze_table(engine)
        print(f"✅ Successfully read {len(df)} records from bronze table!")
        print(f"Columns: {list(df.columns)}")
        return df
    except Exception as e:
        print(f"❌ Bronze data reading error: {e}")
        return None

def test_preprocessing_pipeline():
    """Test the full preprocessing pipeline"""
    print("\nTesting full preprocessing pipeline...")
    try:
        from nlp.preprocessing_nlp_ref.preprocessing_script import (
            create_db_engine,
            process_reviews_data_from_db
        )
        
        engine = create_db_engine()
        processed_df = process_reviews_data_from_db(engine)
        print(f"✅ Successfully processed {len(processed_df)} records!")
        print(f"Processed columns: {list(processed_df.columns)}")
        return processed_df
    except Exception as e:
        print(f"❌ Preprocessing pipeline error: {e}")
        return None

def test_ingest_to_silver():
    """Test ingesting to silver table"""
    print("\nTesting ingestion to silver table...")
    try:
        from nlp.preprocessing_nlp_ref.preprocessing_script import (
            create_db_engine,
            process_reviews_data_from_db,
            ingest_to_silver_table
        )
        
        engine = create_db_engine()
        processed_df = process_reviews_data_from_db(engine)
        
        # Take only first 10 records for testing
        test_df = processed_df.head(10)
        ingest_to_silver_table(test_df, engine)
        print(f"✅ Successfully ingested {len(test_df)} test records to silver table!")
        return True
    except Exception as e:
        print(f"❌ Silver ingestion error: {e}")
        return False

def main():
    """Main test function"""
    print("=" * 60)
    print("TESTING PREPROCESSING TASK")
    print("=" * 60)
    
    # Test 1: Imports
    if not test_preprocessing_imports():
        print("\n❌ Preprocessing tests failed at import stage")
        return
    
    # Test 2: Setup
    engine = test_preprocessing_setup()
    if engine is None:
        print("\n❌ Preprocessing tests failed at setup stage")
        return
    
    # Test 3: Schema and table
    if not test_schema_and_table():
        print("\n❌ Preprocessing tests failed at schema/table stage")
        return
    
    # Test 4: Read bronze data
    df = test_read_bronze_data()
    if df is None:
        print("\n❌ Preprocessing tests failed at bronze reading stage")
        return
    
    # Test 5: Processing pipeline
    processed_df = test_preprocessing_pipeline()
    if processed_df is None:
        print("\n❌ Preprocessing tests failed at processing stage")
        return
    
    # Test 6: Silver ingestion (optional, with small dataset)
    if len(processed_df) > 0:
        test_ingest_to_silver()
    
    print("\n" + "=" * 60)
    print("✅ ALL PREPROCESSING TESTS COMPLETED!")
    print("=" * 60)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3

import sys
import os
sys.path.append('/home/kariem/airflow/src')

def test_preprocessing_imports():
    """Test if we can import preprocessing functions"""
    try:
        print("Testing imports...")
        from nlp.preprocessing_nlp_ref.preprocessing_script import (
            create_db_engine,
            ensure_schema_and_table_exist,
            read_from_bronze_table,
            process_reviews_data_from_db,
            ingest_to_silver_table
        )
        print("✓ All imports successful")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

def test_database_connection():
    """Test database connection creation"""
    try:
        print("Testing database connection...")
        from nlp.preprocessing_nlp_ref.preprocessing_script import create_db_engine
        
        engine = create_db_engine()
        print(f"✓ Engine created: {type(engine)}")
        
        # Test a simple connection
        with engine.connect() as conn:
            result = conn.execute("SELECT 1 as test")
            print(f"✓ Connection test successful: {result.fetchone()}")
        
        engine.dispose()
        return True
    except Exception as e:
        print(f"✗ Database connection failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_schema_setup():
    """Test schema and table setup"""
    try:
        print("Testing schema setup...")
        from nlp.preprocessing_nlp_ref.preprocessing_script import (
            create_db_engine,
            ensure_schema_and_table_exist
        )
        
        engine = create_db_engine()
        ensure_schema_and_table_exist(engine)
        print("✓ Schema setup successful")
        
        engine.dispose()
        return True
    except Exception as e:
        print(f"✗ Schema setup failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_reading():
    """Test reading from bronze table"""
    try:
        print("Testing data reading...")
        from nlp.preprocessing_nlp_ref.preprocessing_script import (
            create_db_engine,
            read_from_bronze_table
        )
        
        engine = create_db_engine()
        df = read_from_bronze_table(engine)
        print(f"✓ Data reading successful: {len(df)} records")
        
        engine.dispose()
        return True
    except Exception as e:
        print(f"✗ Data reading failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Starting preprocessing tests...")
    
    tests = [
        test_preprocessing_imports,
        test_database_connection,
        test_schema_setup,
        test_data_reading
    ]
    
    for test in tests:
        print(f"\n{'='*50}")
        success = test()
        if not success:
            print(f"Test {test.__name__} failed, stopping here.")
            break
    
    print("\nTesting completed!")

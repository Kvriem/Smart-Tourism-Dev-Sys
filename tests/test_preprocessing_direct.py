#!/usr/bin/env python3
"""
Direct test of the preprocessing task function
"""

import sys
import os
import logging

# Add the src directory to the path
sys.path.append('/home/kariem/airflow/src')
sys.path.append('/home/kariem/airflow')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_preprocessing_imports():
    """Test if all preprocessing imports work"""
    try:
        print("Testing preprocessing imports...")
        
        # Test preprocessing script imports
        from nlp.preprocessing_nlp_ref.preprocessing_script import (
            create_db_engine,
            ensure_schema_and_table_exist,
            process_reviews_data_from_db,
            ingest_to_silver_table
        )
        print("✓ Preprocessing script imports successful")
        
        return True
    except Exception as e:
        print(f"✗ Import error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_preprocessing_setup():
    """Test the preprocessing setup function"""
    try:
        print("\nTesting preprocessing setup...")
        
        from nlp.preprocessing_nlp_ref.preprocessing_script import (
            create_db_engine,
            ensure_schema_and_table_exist
        )
        
        # Test database engine creation
        print("Creating database engine...")
        engine = create_db_engine()
        print("✓ Database engine created successfully")
        
        # Test schema and table creation
        print("Ensuring schema and table exist...")
        ensure_schema_and_table_exist(engine)
        print("✓ Schema and table setup completed")
        
        engine.dispose()
        print("✓ Preprocessing setup test completed successfully")
        
        return True
    except Exception as e:
        print(f"✗ Preprocessing setup error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_preprocessing():
    """Test the data preprocessing function"""
    try:
        print("\nTesting data preprocessing...")
        
        from nlp.preprocessing_nlp_ref.preprocessing_script import (
            create_db_engine,
            process_reviews_data_from_db,
            ingest_to_silver_table
        )
        
        # Create engine
        engine = create_db_engine()
        
        # Process reviews data
        print("Processing reviews data...")
        processed_df = process_reviews_data_from_db(engine)
        print(f"✓ Processed {len(processed_df)} records")
        
        # Ingest to silver table
        print("Ingesting to silver table...")
        ingest_to_silver_table(processed_df, engine)
        print("✓ Data ingestion completed")
        
        engine.dispose()
        print("✓ Data preprocessing test completed successfully")
        
        return True
    except Exception as e:
        print(f"✗ Data preprocessing error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("=" * 60)
    print("PREPROCESSING TASK TESTING")
    print("=" * 60)
    
    tests = [
        ("Import Test", test_preprocessing_imports),
        ("Setup Test", test_preprocessing_setup),
        ("Data Processing Test", test_data_preprocessing)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)
    
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name}: {status}")
    
    total_passed = sum(1 for _, passed in results if passed)
    print(f"\nOverall: {total_passed}/{len(results)} tests passed")
    
    if total_passed == len(results):
        print("🎉 All preprocessing tests passed!")
        return 0
    else:
        print("❌ Some preprocessing tests failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())

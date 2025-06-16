#!/usr/bin/env python3
"""
Test script for Preprocessing and Translation tasks
"""
import sys
import os
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Add the src directory to Python path
sys.path.append('/home/kariem/airflow/src')

print("Testing Preprocessing and Translation Tasks...")

def test_preprocessing_setup():
    """Test preprocessing setup task"""
    print("\n=== Testing Preprocessing Setup ===")
    try:
        from nlp.preprocessing_nlp_ref.preprocessing_script import (
            create_db_engine,
            ensure_schema_and_table_exist
        )
        
        engine = create_db_engine()
        ensure_schema_and_table_exist(engine)
        engine.dispose()
        
        print("✅ Preprocessing setup successful")
        return True
    except Exception as e:
        print(f"❌ Preprocessing setup failed: {e}")
        return False

def test_data_preprocessing():
    """Test data preprocessing task"""
    print("\n=== Testing Data Preprocessing ===")
    try:
        from nlp.preprocessing_nlp_ref.preprocessing_script import (
            create_db_engine,
            process_reviews_data_from_db,
            ingest_to_silver_table
        )
        
        engine = create_db_engine()
        
        # Process reviews data
        processed_df = process_reviews_data_from_db(engine)
        
        # Ingest to silver table
        ingest_to_silver_table(processed_df, engine)
        
        engine.dispose()
        
        print(f"✅ Data preprocessing successful. Processed {len(processed_df)} records")
        return True
    except Exception as e:
        print(f"❌ Data preprocessing failed: {e}")
        return False

def test_preprocessing_validation():
    """Test preprocessing validation task"""
    print("\n=== Testing Preprocessing Validation ===")
    try:
        from nlp.preprocessing_nlp_ref.preprocessing_script import create_db_engine
        from sqlalchemy import text
        
        engine = create_db_engine()
        
        # Get count of records in silver table
        with engine.connect() as conn:
            result = conn.execute(text("SELECT COUNT(*) FROM silver.reviews_cleaned"))
            silver_count = result.fetchone()[0]
            
            # Get count of records in bronze table
            result = conn.execute(text("SELECT COUNT(*) FROM bronze.hotels_reviews_test"))
            bronze_count = result.fetchone()[0]
        
        engine.dispose()
        
        print(f"✅ Preprocessing validation successful. Bronze: {bronze_count}, Silver: {silver_count} records")
        return True
    except Exception as e:
        print(f"❌ Preprocessing validation failed: {e}")
        return False

def test_translation_setup():
    """Test translation setup task"""
    print("\n=== Testing Translation Setup ===")
    try:
        from nlp.preprocessing_nlp_ref.translation_local_gpu import (
            create_optimized_engine,
            ensure_table_exists_optimized
        )
        
        engine = create_optimized_engine()
        ensure_table_exists_optimized(engine)
        engine.dispose()
        
        print("✅ Translation setup successful")
        return True
    except Exception as e:
        print(f"❌ Translation setup failed: {e}")
        return False

def test_load_data_for_translation():
    """Test load data for translation task"""
    print("\n=== Testing Load Data for Translation ===")
    try:
        from nlp.preprocessing_nlp_ref.translation_local_gpu import (
            create_optimized_engine,
            read_data_optimized
        )
        
        engine = create_optimized_engine()
        
        # Read data optimized
        df = read_data_optimized(engine)
        
        engine.dispose()
        
        print(f"✅ Load data for translation successful. Loaded {len(df)} records with shape {df.shape}")
        return True
    except Exception as e:
        print(f"❌ Load data for translation failed: {e}")
        return False

def test_translate_reviews():
    """Test translate reviews task (quick test with small sample)"""
    print("\n=== Testing Translate Reviews ===")
    try:
        from nlp.preprocessing_nlp_ref.translation_local_gpu import (
            create_optimized_engine,
            read_data_optimized,
            translate_reviews_ultra_fast,
            perf_cache
        )
        
        engine = create_optimized_engine()
        
        # Read data for translation (limit to first 10 records for testing)
        df = read_data_optimized(engine, limit=10)
        
        # Perform translation
        translated_df = translate_reviews_ultra_fast(df)
        
        engine.dispose()
        
        # Get cache stats
        stats = perf_cache.get_stats()
        
        print(f"✅ Translation successful for {len(translated_df)} records")
        print(f"   Cache hit rate: {stats['hit_rate']:.1f}%")
        return True
    except Exception as e:
        print(f"❌ Translation failed: {e}")
        return False

def test_translation_validation():
    """Test translation validation task"""
    print("\n=== Testing Translation Validation ===")
    try:
        from nlp.preprocessing_nlp_ref.translation_local_gpu import create_optimized_engine
        from sqlalchemy import text
        
        engine = create_optimized_engine()
        
        # Get count of records in translated table
        with engine.connect() as conn:
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
        
        print(f"✅ Translation validation successful:")
        print(f"   Source records: {source_count}")
        print(f"   Translated table records: {translated_count}")
        print(f"   Records with translations: {records_with_translation}")
        
        return True
    except Exception as e:
        print(f"❌ Translation validation failed: {e}")
        return False

def main():
    """Run all tests"""
    print("Starting Preprocessing and Translation Task Tests...")
    
    tests = [
        ("Preprocessing Setup", test_preprocessing_setup),
        ("Data Preprocessing", test_data_preprocessing),
        ("Preprocessing Validation", test_preprocessing_validation),
        ("Translation Setup", test_translation_setup),
        ("Load Data for Translation", test_load_data_for_translation),
        ("Translate Reviews", test_translate_reviews),
        ("Translation Validation", test_translation_validation),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    print("\n" + "="*50)
    print("TEST RESULTS SUMMARY")
    print("="*50)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests passed!")
    else:
        print("⚠️  Some tests failed. Check the errors above.")

if __name__ == "__main__":
    main()

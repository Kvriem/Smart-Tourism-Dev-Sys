#!/usr/bin/env python3
import sys
import os

print("=== Basic Import Test ===")
print(f"Python version: {sys.version}")
print(f"Current working directory: {os.getcwd()}")

# Add src to path
sys.path.insert(0, '/home/kariem/airflow/src')
print(f"Python path includes: {[p for p in sys.path if 'airflow' in p]}")

# Test basic imports
try:
    import pandas as pd
    print("✅ pandas imported successfully")
except Exception as e:
    print(f"❌ pandas failed: {e}")

try:
    import sqlalchemy
    print("✅ sqlalchemy imported successfully") 
except Exception as e:
    print(f"❌ sqlalchemy failed: {e}")

try:
    import nltk
    print("✅ nltk imported successfully")
except Exception as e:
    print(f"❌ nltk failed: {e}")

try:
    from deep_translator import GoogleTranslator
    print("✅ deep_translator imported successfully")
except Exception as e:
    print(f"❌ deep_translator failed: {e}")

# Test our modules
try:
    from nlp.preprocessing_nlp_ref.preprocessing_script import create_db_engine
    print("✅ preprocessing_script imported successfully")
except Exception as e:
    print(f"❌ preprocessing_script failed: {e}")

try:
    from nlp.preprocessing_nlp_ref.translation_local_gpu import create_optimized_engine
    print("✅ translation_local_gpu imported successfully")
except Exception as e:
    print(f"❌ translation_local_gpu failed: {e}")

print("=== Import test completed ===")

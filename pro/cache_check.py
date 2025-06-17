import pickle
import pandas as pd
import os

# Direct cache access
cache_file = os.path.join('cache', 'data_cache.pkl')

if os.path.exists(cache_file):
    try:
        with open(cache_file, 'rb') as f:
            df = pickle.load(f)
        
        print("DATABASE STRUCTURE ANALYSIS")
        print("Total records:", len(df))
        print("Columns:", df.columns.tolist())
        print()
        
        print("SAMPLE DATA (first 2 rows)")
        print(df.head(2).to_string())
        print()
        
        # Check for revenue-related columns
        revenue_keywords = ['revenue', 'price', 'cost', 'spending', 'payment', 'fee', 'rate', 'amount', 'money', 'dollar', 'usd', 'euro']
        revenue_cols = []
        for col in df.columns:
            if any(keyword in col.lower() for keyword in revenue_keywords):
                revenue_cols.append(col)
        
        print("REVENUE-RELATED COLUMNS")
        if revenue_cols:
            print("Found revenue columns:", revenue_cols)
        else:
            print("No revenue-related columns found!")
        print()
        
        print("KEY FINDINGS FOR TOURISM ECONOMICS CHART:")
        print("1. Nationality column available:", 'Nationality' in df.columns)
        print("2. Sentiment column available:", any('sentiment' in col.lower() for col in df.columns))
        print("3. Revenue/spending data available:", len(revenue_cols) > 0)
        print()
        
        if len(revenue_cols) == 0:
            print("CRITICAL ISSUE: No actual revenue data found!")
            print("The economic impact calculations are based on ESTIMATES, not real data.")
            print("This means the 'Economic Impact' values in the chart are simulated.")
        
        # Check investment ties issue
        print()
        print("INVESTMENT TIES ANALYSIS:")
        print("The chart uses predefined spending estimates per nationality:")
        print("- These are hardcoded values, not from the database")
        print("- This explains why 'Investment Ties' appear the same")
        print("- The chart needs real financial data to be accurate")
        
    except Exception as e:
        print("Error reading cache:", str(e))
else:
    print("Cache file not found")

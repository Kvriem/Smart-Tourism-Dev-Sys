#!/usr/bin/env python3
"""
Test script for the scrape_reviews task
"""

import sys
import os
import logging
from datetime import datetime
from unittest.mock import MagicMock

# Add the src directory to Python path
sys.path.append('/home/kariem/airflow/src')
sys.path.append('/home/kariem/airflow/dags')

# Import the task function
from scheduled import task_scrape_hotel_reviews

def create_mock_context():
    """Create a mock Airflow context for testing"""
    mock_ti = MagicMock()
    mock_ti.xcom_push = MagicMock()
    
    context = {
        'task_instance': mock_ti,
        'ds': datetime.now().strftime('%Y-%m-%d'),
        'execution_date': datetime.now(),
        'dag': MagicMock(),
        'task': MagicMock()
    }
    
    return context

def test_scrape_reviews_task():
    """Test the scrape_reviews task"""
    print("=" * 60)
    print("TESTING SCRAPE_REVIEWS TASK")
    print("=" * 60)
    
    try:
        # Create mock context
        context = create_mock_context()
        
        # Run the task
        result = task_scrape_hotel_reviews(**context)
        
        print(f"✅ Task completed successfully!")
        print(f"Result: {result}")
        
        # Print XCom pushes
        print("\n📤 XCom values pushed:")
        for call in context['task_instance'].xcom_push.call_args_list:
            if call.kwargs:
                print(f"  {call.kwargs['key']}: {call.kwargs['value']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Task failed with error: {e}")
        import traceback
        print("\n📋 Full traceback:")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    success = test_scrape_reviews_task()
    
    if success:
        print("\n🎉 Test completed successfully!")
        sys.exit(0)
    else:
        print("\n💥 Test failed!")
        sys.exit(1)

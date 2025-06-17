"""
Cache utilities for the dashboard application.
"""
import os
from datetime import datetime
import json

def clear_cache_metadata():
    """Clear the cache metadata file to force a reload of data"""
    cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache')
    cache_metadata_file = os.path.join(cache_dir, 'cache_metadata.json')
    
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
        print(f"✅ Created cache directory: {cache_dir}")
    
    # Reset metadata to force a reload
    metadata = {
        "last_updated": None,
        "last_id": None,
        "record_count": 0
    }
    
    try:
        with open(cache_metadata_file, 'w') as f:
            json.dump(metadata, f)
        return {"success": True, "message": "Cache metadata cleared, data will be reloaded on next access"}
    except Exception as e:
        return {"success": False, "message": f"Error clearing cache metadata: {e}"}

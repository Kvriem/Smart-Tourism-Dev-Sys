import sys
print("Testing overview import...", flush=True)
try:
    from pages.overview_page import load_data
    print("SUCCESS: Overview imported!", flush=True)
except Exception as e:
    print(f"ERROR: {e}", flush=True)
    import traceback
    traceback.print_exc()
print("Test complete.", flush=True)

#!/usr/bin/env python3
"""
Simple Firefox test script
"""
import subprocess
import sys

def test_firefox():
    try:
        result = subprocess.run(['firefox', '--version'], 
                              capture_output=True, text=True, timeout=10)
        print(f"Firefox test result: {result.returncode}")
        print(f"Firefox output: {result.stdout.strip()}")
        return result.returncode == 0
    except Exception as e:
        print(f"Firefox test error: {e}")
        return False

if __name__ == "__main__":
    print("Testing Firefox...")
    if test_firefox():
        print("✅ Firefox is working!")
    else:
        print("❌ Firefox test failed!")

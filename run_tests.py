#!/usr/bin/env python3
"""
Test runner for the Liar's Dice tournament raw-data capture layer.
"""

import sys
import os
import unittest
import subprocess

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


def run_unit_tests():
    """Run unit tests for the event logging system."""
    print("🧪 Running unit tests...")
    
    # Create test directory if it doesn't exist
    test_dir = os.path.join(os.path.dirname(__file__), 'tests')
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
        # Create empty __init__.py
        with open(os.path.join(test_dir, '__init__.py'), 'w') as f:
            f.write("")
    
    # Discover and run tests
    loader = unittest.TestLoader()
    suite = loader.discover('tests', pattern='test_*.py')
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


def run_smoke_test():
    """Run the smoke test script."""
    print("🚀 Running smoke test...")
    
    try:
        result = subprocess.run([
            sys.executable, 'smoke_test_raw_logging.py'
        ], capture_output=True, text=True)
        
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        return result.returncode == 0
    except Exception as e:
        print(f"Error running smoke test: {e}")
        return False


def main():
    """Main test runner."""
    print("Tournament Raw-Data Capture Layer - Test Suite")
    print("=" * 50)
    
    all_passed = True
    
    # Run unit tests
    unit_tests_passed = run_unit_tests()
    if unit_tests_passed:
        print("✅ Unit tests passed")
    else:
        print("❌ Unit tests failed")
        all_passed = False
    
    print()
    
    # Run smoke test
    smoke_test_passed = run_smoke_test()
    if smoke_test_passed:
        print("✅ Smoke test passed")
    else:
        print("❌ Smoke test failed")
        all_passed = False
    
    print("\n" + "=" * 50)
    
    if all_passed:
        print("🎉 All tests passed!")
        print("\nThe raw-data capture layer is ready for use.")
        print("To enable in tournaments, set raw_logging_enabled=True")
        sys.exit(0)
    else:
        print("💥 Some tests failed!")
        print("Please check the output above for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
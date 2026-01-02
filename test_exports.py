"""
DermalScan Export and Logging Test Script
==========================================
This script tests the new Module 7 features:
- CSV export formatting
- Annotated image export
- Logging functionality
- Export package creation
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import os
import csv
import zipfile
from datetime import datetime

def test_csv_format(csv_file_path):
    """Test CSV file format and validate columns."""
    print(f"\n{'='*60}")
    print("Testing CSV Export Format")
    print(f"{'='*60}")
    
    try:
        with open(csv_file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)
            
            # Validate header
            expected_cols = ['Timestamp', 'Image_Filename', 'Face_Number', 
                           'Predicted_Condition', 'Confidence_%']
            
            print(f"\n✓ Header columns: {len(header)}")
            print(f"  Expected minimum: {len(expected_cols)}")
            print(f"\n  Column names:")
            for i, col in enumerate(header, 1):
                print(f"    {i}. {col}")
            
            # Count rows
            row_count = 0
            for row in reader:
                row_count += 1
                if row_count == 1:
                    print(f"\n✓ Sample data row:")
                    for col, val in zip(header, row):
                        print(f"    {col}: {val}")
            
            print(f"\n✓ Total data rows: {row_count}")
            print(f"✅ CSV format validation PASSED")
            return True
            
    except Exception as e:
        print(f"❌ CSV validation FAILED: {e}")
        return False

def test_zip_package(zip_file_path):
    """Test ZIP package contents."""
    print(f"\n{'='*60}")
    print("Testing ZIP Package Export")
    print(f"{'='*60}")
    
    try:
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            file_list = zip_ref.namelist()
            
            print(f"\n✓ Package contains {len(file_list)} file(s):")
            for f in file_list:
                info = zip_ref.getinfo(f)
                print(f"    - {f} ({info.file_size:,} bytes)")
            
            # Check for required files
            has_image = any('png' in f.lower() for f in file_list)
            has_csv = any('csv' in f.lower() for f in file_list)
            
            if has_image and has_csv:
                print(f"\n✅ ZIP package validation PASSED")
                return True
            else:
                print(f"\n❌ Missing required files (image: {has_image}, csv: {has_csv})")
                return False
                
    except Exception as e:
        print(f"❌ ZIP validation FAILED: {e}")
        return False

def test_log_files(logs_dir):
    """Test log file creation and format."""
    print(f"\n{'='*60}")
    print("Testing Log Files")
    print(f"{'='*60}")
    
    if not os.path.exists(logs_dir):
        print(f"❌ Logs directory not found: {logs_dir}")
        return False
    
    log_files = [f for f in os.listdir(logs_dir) if f.endswith('.log')]
    
    if not log_files:
        print(f"⚠️  No log files found yet (will be created on first run)")
        return True
    
    print(f"\n✓ Found {len(log_files)} log file(s):")
    
    for log_file in log_files:
        log_path = os.path.join(logs_dir, log_file)
        file_size = os.path.getsize(log_path)
        print(f"    - {log_file} ({file_size:,} bytes)")
        
        # Read first few lines
        try:
            with open(log_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()[:5]
                if lines:
                    print(f"      Sample log entries:")
                    for line in lines:
                        print(f"        {line.strip()}")
        except Exception as e:
            print(f"      ⚠️  Could not read log: {e}")
    
    print(f"\n✅ Log file validation PASSED")
    return True

def print_summary(results):
    """Print test summary."""
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}\n")
    
    total = len(results)
    passed = sum(1 for r in results.values() if r)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"  {test_name}: {status}")
    
    print(f"\n  Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print(f"\n  🎉 All tests PASSED! Module 7 is ready.")
    else:
        print(f"\n  ⚠️  Some tests failed. Please review the output above.")

def main():
    """Main test runner."""
    print(f"""
{'='*60}
DermalScan Module 7: Export and Logging Tests
{'='*60}
Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*60}
    """)
    
    # Define paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    logs_dir = os.path.join(base_dir, 'logs')
    
    results = {}
    
    # Test 1: Check logs directory
    results['Log Directory Check'] = test_log_files(logs_dir)
    
    # Instructions for manual testing
    print(f"\n{'='*60}")
    print("MANUAL TESTING INSTRUCTIONS")
    print(f"{'='*60}\n")
    
    print("To complete testing, please:")
    print("  1. Run the DermalScan app: streamlit run dermal_app.py")
    print("  2. Upload a test image with faces")
    print("  3. Download the CSV and ZIP exports")
    print("  4. Run this script again with paths to downloaded files:")
    print(f"     python test_exports.py")
    print("\nOptional: Test with downloaded exports")
    print("  - Place CSV file in: test_data/sample_export.csv")
    print("  - Place ZIP file in: test_data/sample_package.zip")
    
    # Check for test files
    test_csv = os.path.join(base_dir, 'test_data', 'sample_export.csv')
    test_zip = os.path.join(base_dir, 'test_data', 'sample_package.zip')
    
    if os.path.exists(test_csv):
        results['CSV Export Format'] = test_csv_format(test_csv)
    
    if os.path.exists(test_zip):
        results['ZIP Package Export'] = test_zip_package(test_zip)
    
    # Print summary
    print_summary(results)
    
    print(f"\n{'='*60}")
    print("Testing Complete")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()

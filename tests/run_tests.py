#!/usr/bin/env python3
"""
Test Runner Script for AI Agriculture Yield Production

This script provides an easy way to run the complete test suite
for the preprocessing functions.

Usage:
    python run_tests.py                    # Run all tests
    python run_tests.py --basic            # Run basic tests only
    python run_tests.py --comprehensive    # Run comprehensive tests only
    python run_tests.py --coverage         # Run with coverage report
    python run_tests.py --help             # Show help
"""

import subprocess
import sys
import argparse
import os


def run_command(command, description):
    """Run a shell command and return the result."""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        return result.returncode == 0
    except Exception as e:
        print(f"Error running command: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Test Runner for AI Agriculture Yield Production",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_tests.py                    # Run all tests
    python run_tests.py --basic            # Run basic tests only
    python run_tests.py --comprehensive    # Run comprehensive tests only
    python run_tests.py --coverage         # Run with coverage report
        """
    )
    
    parser.add_argument("--basic", action="store_true", 
                       help="Run basic functionality tests only")
    parser.add_argument("--comprehensive", action="store_true", 
                       help="Run comprehensive tests only")
    parser.add_argument("--coverage", action="store_true", 
                       help="Run tests with coverage report")
    parser.add_argument("--verbose", "-v", action="store_true", 
                       help="Verbose output")
    
    args = parser.parse_args()
    
    # Check if we're in the right directory
    if not os.path.exists("tests") or not os.path.exists("src"):
        print("Error: Please run this script from the project root directory")
        print("   Make sure both 'tests' and 'src' directories exist.")
        sys.exit(1)
    
    # Check if virtual environment is activated
    venv_python = ".venv/bin/python"
    if not os.path.exists(venv_python):
        print("Error: Virtual environment not found")
        print("   Please ensure .venv is set up and activated")
        sys.exit(1)
    
    print("AI Agriculture Yield Production - Test Suite Runner")
    print("Comprehensive Testing")
    
    base_cmd = f"{venv_python} -m pytest"
    verbose_flag = " -v" if args.verbose else ""
    
    success = True
    
    if args.basic:
        cmd = f"{base_cmd} tests/test_preprocessing_simple.py{verbose_flag}"
        success &= run_command(cmd, "Running Basic Functionality Tests")
        
    elif args.comprehensive:
        cmd = f"{base_cmd} tests/test_data_preprocessing_comprehensive.py{verbose_flag}"
        success &= run_command(cmd, "Running Comprehensive Test Suite")
        
    elif args.coverage:
        cmd = f"{base_cmd} tests/ --cov=src --cov-report=html --cov-report=term{verbose_flag}"
        success &= run_command(cmd, "Running Tests with Coverage Analysis")
        print("\nCoverage report generated in htmlcov/index.html")
        
    else:
        # Run all tests by default
        cmd = f"{base_cmd} tests/{verbose_flag}"
        success &= run_command(cmd, "Running Complete Test Suite")
    
    # Summary
    print(f"\n{'='*60}")
    if success:
        print("All tests completed successfully!")
        print("AI Agriculture preprocessing tests PASSED")
    else:
        print("Some tests failed. Please check the output above.")
        sys.exit(1)
    
    print(f"{'='*60}")
    print("\nAvailable test categories:")
    print("   • Data Loading and File Handling")
    print("   • Missing Value Detection and Handling")
    print("   • Outlier Detection and Removal")
    print("   • Feature Encoding and Scaling")
    print("   • Data Splitting and Validation")
    print("   • Integration Pipeline Testing")
    print("\nSee tests/README.md for detailed documentation")


if __name__ == "__main__":
    main()

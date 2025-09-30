#!/usr/bin/env python3
"""
Demo script for background subtraction fish tracking.

This script demonstrates how to use the new background subtraction
tracking method for 28 fish.
"""

import sys
import os
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tracking_methods.background_subtraction.tracking_program_28fish_bg_subtraction import main as run_bg_subtraction_tracking
from tracking_methods.traditional_tracking.tracking_program_28fish_traditional import main as run_traditional_tracking


def main():
    """Run the background subtraction tracking demo."""
    print("=" * 60)
    print("Fish Detection - Background Subtraction Tracking Demo")
    print("=" * 60)
    print()
    print("This demo shows how to use the new background subtraction")
    print("tracking method for 28 fish.")
    print()
    print("Available tracking methods:")
    print("1. Background Subtraction (NEW)")
    print("2. Traditional Method (Updated for 28 fish)")
    print()
    
    choice = input("Select tracking method (1 or 2): ").strip()
    
    if choice == "1":
        print("\nRunning background subtraction tracking for 28 fish...")
        print("This method uses your custom background subtraction implementation.")
        try:
            run_bg_subtraction_tracking()
            print("\n✅ Background subtraction tracking completed successfully!")
        except Exception as e:
            print(f"\n❌ Error running background subtraction tracking: {e}")
    
    elif choice == "2":
        print("\nRunning traditional tracking for 28 fish...")
        print("This method uses the original tracking approach (updated for 28 fish).")
        try:
            run_traditional_tracking()
            print("\n✅ Traditional tracking completed successfully!")
        except Exception as e:
            print(f"\n❌ Error running traditional tracking: {e}")
    
    else:
        print("Invalid choice. Please run the script again and select 1 or 2.")
        return
    
    print("\n" + "=" * 60)
    print("Demo completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()


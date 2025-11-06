"""
Script to help download the Goodbooks-10k dataset
"""
import os
import sys

def main():
    if sys.platform == 'win32':
        sys.stdout.reconfigure(encoding='utf-8')
    
    print("Goodbooks-10k Dataset Download Instructions")
    print("\nDataset source: https://github.com/zygmuntz/goodbooks-10k")
    print("\nRequired file: books.csv")
    print(f"Target location: {os.path.abspath('data/goodbooks-10k/books.csv')}")
    
    print("\nDownload steps:")
    print("1. Visit: https://github.com/zygmuntz/goodbooks-10k")
    print("2. Click 'Code' -> 'Download ZIP'")
    print("3. Extract books.csv from the archive")
    print("4. Place books.csv in data/goodbooks-10k/ directory")
    
    if os.path.exists('data/goodbooks-10k/books.csv'):
        print("\nStatus: Dataset found")
        print("Next step: python test_recommender.py")
    else:
        print("\nStatus: Dataset not found")
        print("Action required: Download and place books.csv in correct location")

if __name__ == "__main__":
    main()

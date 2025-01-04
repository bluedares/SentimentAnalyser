from tabulate import tabulate
from colorama import Fore
import os

def format_results(results: dict) -> None:
    """Format and display the sentiment analysis results"""
    print(Fore.CYAN + "\n=== Analysis Results ===")
    print(Fore.WHITE + f"\nReview: {results['review']}\n")

    # Prepare data for tabulation
    headers = ["Model", "Analysis"]
    table_data = []
    
    for analysis in results['analyses']:
        table_data.append([
            analysis['model'],
            analysis['analysis'].replace('\n', ' ')
        ])
    
    # Print formatted table
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    print("\n")

import os
from dotenv import load_dotenv
from sentiment_analyzer import SentimentAnalyzer, ModelType
from colorama import init, Fore

# Initialize colorama for colored output
init()

def get_model_choice() -> ModelType:
    while True:
        print(f"\n{Fore.CYAN}Choose the AI model to use:{Fore.WHITE}")
        print("1. OpenAI (GPT-3.5)")
        print("2. Google Gemini")
        print("3. Both models")
        
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == "1":
            return ModelType.OPENAI
        elif choice == "2":
            return ModelType.GEMINI
        elif choice == "3":
            return ModelType.BOTH
        else:
            print(f"{Fore.RED}Invalid choice. Please enter 1, 2, or 3.{Fore.WHITE}")

def main():
    # Load environment variables
    load_dotenv()
    
    # Check for API keys
    if not os.getenv('OPENAI_API_KEY') or not os.getenv('GOOGLE_API_KEY'):
        print(Fore.RED + "Error: Please set OPENAI_API_KEY and GOOGLE_API_KEY in your .env file")
        return

    # Get model choice from user
    model_type = get_model_choice()
    
    # Initialize the analyzer with chosen model
    analyzer = SentimentAnalyzer(model_type)

    print(Fore.CYAN + "\nApp Store Review Sentiment Analyzer")
    print(Fore.WHITE + "Enter your reviews below. Type 'DONE' on a new line when finished:")
    
    # Collect reviews
    reviews = []
    while True:
        line = input()
        if line.strip().upper() == 'DONE':
            break
        reviews.append(line)
    
    if not reviews:
        print(Fore.RED + "No reviews provided.")
        return
    
    # Analyze the reviews
    try:
        results_df = analyzer.analyze_reviews('\n'.join(reviews))
        analyzer.display_analysis(results_df)
    except Exception as e:
        print(Fore.RED + f"An error occurred during analysis: {str(e)}")

if __name__ == "__main__":
    main()

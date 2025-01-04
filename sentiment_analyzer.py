from langchain_openai import ChatOpenAI
import google.generativeai as genai
import os
from typing import Dict, Any, List
from models.review import Review
import pandas as pd
from tabulate import tabulate
from colorama import Fore
from tqdm import tqdm
import time
import threading
import itertools
import sys
from enum import Enum
from rich.console import Console
from rich.table import Table

class ModelType(Enum):
    OPENAI = "openai"
    GEMINI = "gemini"
    BOTH = "both"

class LoadingSpinner:
    def __init__(self, message="Processing"):
        self.message = message
        self.spinner_chars = itertools.cycle(['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'])
        self.running = False
        self.spinner_thread = None

    def spin(self):
        while self.running:
            sys.stdout.write(f'\r{self.message} {next(self.spinner_chars)}')
            sys.stdout.flush()
            time.sleep(0.1)
        sys.stdout.write('\r' + ' ' * (len(self.message) + 2) + '\r')
        sys.stdout.flush()

    def start(self):
        self.running = True
        self.spinner_thread = threading.Thread(target=self.spin)
        self.spinner_thread.start()

    def stop(self):
        self.running = False
        if self.spinner_thread:
            self.spinner_thread.join()

class SentimentAnalyzer:
    def __init__(self, model_type: ModelType = ModelType.BOTH):
        """Initialize the analyzer with specified model type"""
        self.model_type = model_type
        
        # Initialize models based on selection
        if model_type in [ModelType.OPENAI, ModelType.BOTH]:
            self.openai_model = ChatOpenAI(
                model_name="gpt-3.5-turbo",
                temperature=0
            )
        
        if model_type in [ModelType.GEMINI, ModelType.BOTH]:
            genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
            self.gemini_model = genai.GenerativeModel('gemini-pro')
        
        # Define the sentiment analysis prompt template
        self.prompt_template = """Analyze this app store review and provide a JSON-formatted response with:
        1. sentiment: (Positive/Negative/Neutral)
        2. confidence_score: (0-100)
        3. key_aspects: [list of key features/issues mentioned]
        4. main_emotion: (primary emotion expressed)
        5. summary: (brief one-line summary)
        
        Review Text: {review}"""

    def analyze_with_openai(self, review: Review) -> Dict[str, Any]:
        """Analyze sentiment using OpenAI's model"""
        response = self.openai_model.invoke(
            self.prompt_template.format(review=review.text)
        )
        return {
            "model": "ChatGPT",
            "analysis": response.content,
            "stars": review.stars,
            "author": review.author,
            "date": review.date,
            "device": review.device,
            "manufacturer": review.manufacturer,
            "version": review.version,
            "text": review.text,
            "client_id": review.client_id
        }

    def analyze_with_gemini(self, review: Review) -> Dict[str, Any]:
        """Analyze sentiment using Google's Gemini model"""
        response = self.gemini_model.generate_content(
            self.prompt_template.format(review=review.text)
        )
        
        return {
            "model": "Gemini",
            "analysis": response.text,
            "stars": review.stars,
            "author": review.author,
            "date": review.date,
            "device": review.device,
            "manufacturer": review.manufacturer,
            "version": review.version,
            "text": review.text,
            "client_id": review.client_id
        }

    def extract_emotion(self, analysis_text: str) -> str:
        """Extract emotion from analysis text"""
        try:
            emotion_line = [line for line in analysis_text.split('\n') if 'main_emotion' in line.lower()][0]
            emotion = emotion_line.split(':')[1].strip().strip('",').lower()
            # Capitalize first letter
            return emotion.capitalize()
        except:
            return "Unknown"

    def analyze_reviews(self, reviews_text: str) -> pd.DataFrame:
        """Analyze multiple reviews and return a DataFrame with results"""
        # Parse reviews
        reviews = []
        current_review = []
        
        print(f"{Fore.CYAN}Parsing reviews...{Fore.WHITE}")
        for line in reviews_text.split('\n'):
            if line.strip():
                current_review.append(line)
            elif current_review:
                reviews.append(Review.parse_review('\n'.join(current_review)))
                current_review = []
        
        if current_review:
            reviews.append(Review.parse_review('\n'.join(current_review)))

        # Analyze each review
        results = []
        spinner = LoadingSpinner("Analyzing reviews")
        spinner.start()
        
        try:
            for review in reviews:
                try:
                    # Get analysis based on selected model
                    if self.model_type in [ModelType.OPENAI, ModelType.BOTH]:
                        openai_analysis = self.analyze_with_openai(review)
                        results.append(openai_analysis)
                    
                    if self.model_type in [ModelType.GEMINI, ModelType.BOTH]:
                        gemini_analysis = self.analyze_with_gemini(review)
                        results.append(gemini_analysis)
                        
                except Exception as e:
                    print(f"{Fore.RED}Error analyzing review from {review.author}: {str(e)}{Fore.WHITE}")
        finally:
            spinner.stop()

        # Convert to DataFrame
        print(f"{Fore.GREEN}Analysis complete!{Fore.WHITE}")
        df = pd.DataFrame(results)
        return df

    def save_analysis_results(self, content: str, filename: str = "review_analysis.txt", mode: str = 'w') -> None:
        """Save the analysis content to a file"""
        try:
            with open(filename, mode, encoding='utf-8') as f:
                f.write(content)
            if mode == 'w':
                print(f"\n{Fore.GREEN}Analysis results saved to {filename}{Fore.WHITE}")
        except Exception as e:
            print(f"\n{Fore.RED}Error saving to file: {str(e)}{Fore.WHITE}")

    def format_table_dual(self, data, headers):
        """
        Format data into both display (grid) and copy-paste (tab-separated) formats.
        Returns both versions for flexible output.
        """
        if not data:
            return {"display": "", "copy": ""}
        
        # Calculate column widths for display version
        widths = []
        for i, header in enumerate(headers):
            column_data = [str(row[i]) for row in data]
            widths.append(max(len(header), max(len(str(x)) for x in column_data)))
        
        # Create display version (grid format)
        display_lines = []
        separator = "+".join("-" * (w + 2) for w in widths)
        separator = "+" + separator + "+"
        
        # Add headers
        header_row = "|"
        for i, header in enumerate(headers):
            header_row += f" {header:{widths[i]}} |"
        
        display_lines.extend([separator, header_row, separator])
        
        # Add data rows
        for row in data:
            data_row = "|"
            for i, cell in enumerate(row):
                data_row += f" {str(cell):{widths[i]}} |"
            display_lines.append(data_row)
        
        display_lines.append(separator)
        
        # Create copy-paste version (tab-separated)
        copy_lines = []
        copy_lines.append("\t".join(headers))
        for row in data:
            copy_lines.append("\t".join(str(cell) for cell in row))
        
        return {
            "display": "\n".join(display_lines),
            "copy": "\n".join(copy_lines)
        }

    def display_analysis(self, df: pd.DataFrame) -> None:
        """Display the analysis results in a formatted table"""
        # Initialize content for file
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        file_content = [f"Analysis Time: {timestamp}\n"]
        
        # Sentiment Analysis Results (console only)
        print(f"\n{Fore.CYAN}=== Sentiment Analysis Results ==={Fore.WHITE}\n")
        
        # Format the sentiment analysis table
        sentiment_data = []
        for _, row in df.iterrows():
            sentiment_data.append([
                row['author'],
                row['stars'],
                row['model'],
                row['analysis']
            ])
        
        sentiment_tables = self.format_table_dual(
            sentiment_data,
            ['Author', 'Stars', 'Model', 'Analysis']
        )
        # Only print to console, don't add to file
        print(sentiment_tables['display'])
        
        # Process emotions
        print(f"\n{Fore.CYAN}Processing emotions...{Fore.WHITE}")
        emotions = []
        emotion_by_review = {}
        
        for _, row in df.iterrows():
            try:
                emotion = self.extract_emotion(row['analysis'])
                emotions.append(emotion.lower())
                review_key = (row['author'], row['text'])
                emotion_by_review[review_key] = emotion
            except:
                continue
        
        # Count emotions
        emotion_counts = {}
        for emotion in emotions:
            if emotion in ['satisfaction', 'pleased', 'content', 'satisfied']:
                base_emotion = 'satisfaction'
            elif emotion in ['happiness', 'joy', 'delight', 'happy']:
                base_emotion = 'happiness'
            elif emotion in ['appreciation', 'gratitude', 'thankful']:
                base_emotion = 'appreciation'
            elif emotion in ['neutral', 'indifferent']:
                base_emotion = 'neutral'
            elif emotion in ['excitement', 'enthusiastic', 'excited']:
                base_emotion = 'excitement'
            else:
                base_emotion = emotion
            
            emotion_counts[base_emotion] = emotion_counts.get(base_emotion, 0) + 1
        
        # Display emotion summary
        if emotion_counts:
            # Console display with decoration
            print(f"\n{Fore.CYAN}=== Emotion Summary ==={Fore.WHITE}")
            
            # File content with simple title
            file_content.append("\nEmotion Distribution:\n")
            
            emotion_data = [[emotion.capitalize(), count] for emotion, count in 
                          sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True)]
            
            emotion_tables = self.format_table_dual(
                emotion_data,
                ['Emotion', 'Count']
            )
            print(emotion_tables['display'])
            file_content.append(emotion_tables['copy'])
        
        # Display user details
        print(f"\n{Fore.CYAN}=== User Details ==={Fore.WHITE}")
        file_content.append("\nUser Information:\n")
        
        user_data = []
        for _, row in df.iterrows():
            client_id = row.get('client_id', '')
            if client_id:
                client_id = f" (Client ID: {client_id})"
            
            emotion = self.extract_emotion(row['analysis'])
            
            user_data.append([
                row['author'],
                row['device'],
                client_id,
                emotion,
                row['text'][:100] + ('...' if len(row['text']) > 100 else '')
            ])
        
        user_tables = self.format_table_dual(
            user_data,
            ['Name', 'Device', 'Client ID', 'Emotion', 'Review Preview']
        )
        print(user_tables['display'])
        file_content.append(user_tables['copy'])
        
        # Get unique reviews and stats
        unique_reviews = df.groupby(['author', 'text']).agg({
            'stars': 'first',
            'device': 'first',
            'client_id': 'first',
            'manufacturer': 'first'
        }).reset_index()
        
        total_reviews = len(unique_reviews)
        
        # Print summary statistics
        print(f"\n{Fore.CYAN}=== Summary Statistics ==={Fore.WHITE}")
        file_content.append("\nSummary:\n")
        stats = f"Number of Reviews: {total_reviews}\nAverage Rating: {df['stars'].mean():.2f} stars"
        print(stats)
        file_content.append(stats)
        
        # Generate copy-paste friendly table
        print(f"\n{Fore.CYAN}=== Copy-Paste Friendly Format ==={Fore.WHITE}")
        file_content.append("\nDetailed Review Analysis:\n")
        
        friendly_data = []
        for _, row in unique_reviews.iterrows():
            review_key = (row['author'], row['text'])
            emotion = emotion_by_review.get(review_key, "Unknown")
            
            review_text = row['text'].strip()
            if len(review_text) > 50:
                review_text = review_text[:47] + "..."
            
            friendly_data.append([
                str(row['author']),
                emotion,
                f"{row['stars']} stars",
                row['device'],
                review_text
            ])
        
        friendly_tables = self.format_table_dual(
            friendly_data,
            ['User', 'Emotion', 'Rating', 'Device', 'Review']
        )
        print(friendly_tables['display'])
        file_content.append(friendly_tables['copy'])
        
        # Save all content to file
        self.save_analysis_results('\n'.join(file_content))

    def printTable(self, table, align="LLLL", hasHeader=True):
        """Print formatted table with custom alignment
        align: string of L/R/C chars for Left/Right/Center alignment of each column
        hasHeader: whether first row is a header (will be separated by a line)
        """
        # Find maximum width of each column
        cols = len(table[0])
        widths = []
        for col in range(cols):
            width = max(len(str(row[col])) for row in table)
            widths.append(width)
        
        # Create format string based on alignment
        formats = []
        for i, a in enumerate(align):
            if a == 'L':
                formats.append('{:<' + str(widths[i]) + '}')
            elif a == 'R':
                formats.append('{:>' + str(widths[i]) + '}')
            else:  # Center
                formats.append('{:^' + str(widths[i]) + '}')
        
        # Create separator lines
        separator = '┌' + '┬'.join('─' * w for w in widths) + '┐'
        separator_mid = '├' + '┼'.join('─' * w for w in widths) + '┤'
        separator_bottom = '└' + '┴'.join('─' * w for w in widths) + '┘'
        
        # Print table
        result = []
        result.append(separator)
        
        for i, row in enumerate(table):
            formatted_row = []
            for j, cell in enumerate(row):
                formatted_row.append(formats[j].format(str(cell)))
            result.append('│' + '│'.join(formatted_row) + '│')
            if i == 0 and hasHeader:
                result.append(separator_mid)
        
        result.append(separator_bottom)
        return '\n'.join(result)

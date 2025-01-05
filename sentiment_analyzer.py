from langchain_openai import ChatOpenAI
import google.generativeai as genai
import os
from typing import Dict, Any, List, Optional
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
import json
import random

class ModelType(Enum):
    OPENAI = "openai"
    GEMINI = "gemini"
    BOTH = "both"

class Emotion(Enum):
    SATISFACTION = "Satisfaction"
    HAPPINESS = "Happiness"
    APPRECIATION = "Appreciation"
    EXCITEMENT = "Excitement"
    NEUTRAL = "Neutral"
    DISAPPOINTMENT = "Disappointment"
    FRUSTRATION = "Frustration"
    ANGER = "Anger"
    CONFUSION = "Confusion"

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
        
        # Emotion mappings
        self.emotion_mappings = {
            # Satisfaction group
            'satisfied': Emotion.SATISFACTION,
            'content': Emotion.SATISFACTION,
            'pleased': Emotion.SATISFACTION,
            'satisfaction': Emotion.SATISFACTION,
            
            # Happiness group
            'happy': Emotion.HAPPINESS,
            'joy': Emotion.HAPPINESS,
            'delight': Emotion.HAPPINESS,
            'happiness': Emotion.HAPPINESS,
            'cheerful': Emotion.HAPPINESS,
            
            # Appreciation group
            'grateful': Emotion.APPRECIATION,
            'thankful': Emotion.APPRECIATION,
            'appreciative': Emotion.APPRECIATION,
            'appreciation': Emotion.APPRECIATION,
            
            # Excitement group
            'excited': Emotion.EXCITEMENT,
            'enthusiastic': Emotion.EXCITEMENT,
            'thrilled': Emotion.EXCITEMENT,
            'excitement': Emotion.EXCITEMENT,
            
            # Neutral group
            'neutral': Emotion.NEUTRAL,
            'indifferent': Emotion.NEUTRAL,
            'okay': Emotion.NEUTRAL,
            'moderate': Emotion.NEUTRAL,
            
            # Disappointment group
            'disappointed': Emotion.DISAPPOINTMENT,
            'letdown': Emotion.DISAPPOINTMENT,
            'dissatisfied': Emotion.DISAPPOINTMENT,
            'disappointment': Emotion.DISAPPOINTMENT,
            
            # Frustration group
            'frustrated': Emotion.FRUSTRATION,
            'annoyed': Emotion.FRUSTRATION,
            'irritated': Emotion.FRUSTRATION,
            'frustration': Emotion.FRUSTRATION,
            
            # Anger group
            'angry': Emotion.ANGER,
            'upset': Emotion.ANGER,
            'mad': Emotion.ANGER,
            'anger': Emotion.ANGER,
            
            # Confusion group
            'confused': Emotion.CONFUSION,
            'uncertain': Emotion.CONFUSION,
            'unsure': Emotion.CONFUSION,
            'confusion': Emotion.CONFUSION,
        }
        
        # Define the sentiment analysis prompt template
        emotions_list = ", ".join([e.value for e in Emotion])
        self.prompt_template = """Analyze the emotional content of this review and provide a brief summary.
        Choose EXACTLY ONE emotion from this list: {}
        
        Guidelines:
        1. Select the most appropriate emotion from the provided list
        2. Do not use any emotions outside this list
        3. Provide a brief explanation for the chosen emotion
        
        Review Text: {{}}""".format(emotions_list)
        
        # Rate limiting settings
        self.max_retries = 5
        self.initial_delay = 1  # Initial delay in seconds
        self.max_delay = 32     # Maximum delay in seconds
        self.batch_size = 5     # Number of reviews to process in parallel
        self.delay_between_batches = 2  # Delay between batches in seconds

    def _handle_api_error(self, e: Exception, attempt: int, review_author: str) -> Optional[str]:
        """Handle API errors with exponential backoff"""
        if attempt >= self.max_retries:
            print(f"Error analyzing review from {review_author}: Maximum retries exceeded")
            return None
            
        if hasattr(e, 'status_code'):
            if e.status_code == 429:  # Rate limit exceeded
                delay = min(self.initial_delay * (2 ** attempt) + random.uniform(0, 1), self.max_delay)
                print(f"\nRate limit hit, waiting {delay:.2f} seconds before retry...")
                time.sleep(delay)
                return None
            elif e.status_code >= 500:  # Server error
                delay = min(self.initial_delay * (2 ** attempt), self.max_delay)
                time.sleep(delay)
                return None
                
        print(f"Error analyzing review from {review_author}: {str(e)}")
        return None

    def _analyze_with_openai(self, review_text: str, review_author: str, attempt: int = 0) -> Optional[str]:
        """Analyze review using OpenAI with retry logic"""
        try:
            response = self.openai_model.invoke(
                self.prompt_template.format(review_text)
            )
            return response.content
        except Exception as e:
            result = self._handle_api_error(e, attempt, review_author)
            if result is None and attempt < self.max_retries:
                return self._analyze_with_openai(review_text, review_author, attempt + 1)
            return result

    def _analyze_with_gemini(self, review_text: str, review_author: str, attempt: int = 0) -> Optional[str]:
        """Analyze review using Gemini with retry logic"""
        try:
            response = self.gemini_model.generate_content(
                self.prompt_template.format(review_text)
            )
            return response.text
        except Exception as e:
            result = self._handle_api_error(e, attempt, review_author)
            if result is None and attempt < self.max_retries:
                return self._analyze_with_gemini(review_text, review_author, attempt + 1)
            return result

    def analyze_with_openai(self, review: Review) -> Dict[str, Any]:
        """Analyze sentiment using OpenAI's model"""
        response = self._analyze_with_openai(review.text, review.author)
        return {
            "model": "ChatGPT",
            "analysis": response,
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
        response = self._analyze_with_gemini(review.text, review.author)
        return {
            "model": "Gemini",
            "analysis": response,
            "stars": review.stars,
            "author": review.author,
            "date": review.date,
            "device": review.device,
            "manufacturer": review.manufacturer,
            "version": review.version,
            "text": review.text,
            "client_id": review.client_id
        }

    def map_to_predefined_emotion(self, text: str) -> Emotion:
        """Map any emotion text to our predefined set of emotions"""
        # Extract the first word (assumed to be the emotion)
        words = text.lower().split()
        if not words:
            return Emotion.NEUTRAL
            
        # Try to find a match in our mappings
        for word in words:
            if word in self.emotion_mappings:
                return self.emotion_mappings[word]
            
            # Try to find partial matches
            for emotion_word, emotion in self.emotion_mappings.items():
                if emotion_word in word or word in emotion_word:
                    return emotion
        
        # Default to neutral if no match found
        return Emotion.NEUTRAL

    def extract_emotion(self, analysis: str) -> str:
        """Extract emotion from analysis text and map to predefined emotion"""
        try:
            # Get the first line or sentence which should contain the emotion
            first_line = analysis.split('\n')[0].split('.')[0]
            mapped_emotion = self.map_to_predefined_emotion(first_line)
            return mapped_emotion.value
        except:
            return Emotion.NEUTRAL.value

    def determine_sentiment(self, stars: int, emotion: str) -> str:
        """Determine sentiment based on stars and emotion"""
        # Map emotions to sentiments
        positive_emotions = [
            Emotion.SATISFACTION.value,
            Emotion.HAPPINESS.value,
            Emotion.APPRECIATION.value,
            Emotion.EXCITEMENT.value
        ]
        
        negative_emotions = [
            Emotion.DISAPPOINTMENT.value,
            Emotion.FRUSTRATION.value,
            Emotion.ANGER.value
        ]
        
        # Determine by stars first
        if stars >= 4:
            return "Positive"
        elif stars <= 2:
            return "Negative"
        
        # For 3 stars, use emotion to determine
        if emotion in positive_emotions:
            return "Positive"
        elif emotion in negative_emotions:
            return "Negative"
        return "Neutral"

    def analyze_reviews(self, reviews_text: str) -> pd.DataFrame:
        """Analyze multiple reviews and return a DataFrame with results"""
        # Parse reviews
        print(f"{Fore.CYAN}Parsing reviews...{Fore.WHITE}")
        reviews = []
        current_review = []
        
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
        
        # Process emotions and sentiments
        print(f"\n{Fore.CYAN}Processing emotions...{Fore.WHITE}")
        emotions = []
        emotion_by_review = {}
        sentiment_by_review = {}
        sentiment_counts = {'Positive': 0, 'Negative': 0, 'Neutral': 0}
        
        for _, row in df.iterrows():
            try:
                emotion = self.extract_emotion(row['analysis'])
                emotions.append(emotion.lower())
                review_key = (row['author'], row['text'])
                emotion_by_review[review_key] = emotion
                sentiment = self.determine_sentiment(row['stars'], emotion)
                sentiment_by_review[review_key] = sentiment
                sentiment_counts[sentiment] += 1
            except:
                continue
        
        # Display sentiment summary
        print(f"\n{Fore.CYAN}=== Sentiment Summary ==={Fore.WHITE}")
        file_content.append("\nSentiment Distribution:\n")
        
        total_reviews = sum(sentiment_counts.values())
        sentiment_data = []
        for sentiment, count in sentiment_counts.items():
            percentage = (count / total_reviews * 100) if total_reviews > 0 else 0
            sentiment_data.append([
                sentiment,
                count,
                f"{percentage:.1f}%"
            ])
        
        sentiment_tables = self.format_table_dual(
            sentiment_data,
            ['Sentiment', 'Count', 'Percentage']
        )
        print(sentiment_tables['display'])
        file_content.append(sentiment_tables['copy'])
        
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
            
            emotion_data = []
            for emotion, count in sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True):
                # Get sentiment for this emotion using a sample review
                sample_review = next((key for key, val in emotion_by_review.items() if val.lower() == emotion.lower()), None)
                sentiment = sentiment_by_review.get(sample_review, "Neutral")
                
                emotion_data.append([
                    emotion.capitalize(),
                    count,
                    sentiment
                ])
            
            emotion_tables = self.format_table_dual(
                emotion_data,
                ['Emotion', 'Count', 'Sentiment']
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
            review_key = (row['author'], row['text'])
            sentiment = sentiment_by_review.get(review_key, "Neutral")
            
            user_data.append([
                row['author'],
                row['device'],
                client_id,
                emotion,
                sentiment,
                row['stars'],
                row['text'][:100] + ('...' if len(row['text']) > 100 else '')
            ])
        
        user_tables = self.format_table_dual(
            user_data,
            ['Name', 'Device', 'Client ID', 'Emotion', 'Sentiment', 'Rating', 'Review Preview']
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

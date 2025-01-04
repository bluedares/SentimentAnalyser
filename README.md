# Sentiment Analyzer

A powerful sentiment analysis tool that processes user reviews using multiple language models (OpenAI GPT and Google Gemini) to provide detailed emotional insights and analysis.

## Features

- **Multi-Model Analysis**: Choose between OpenAI GPT-3.5, Google Gemini, or both for sentiment analysis
- **Emotion Extraction**: Automatically identifies and categorizes emotions from reviews
- **Detailed Analytics**: Provides comprehensive analysis including:
  - Emotion distribution
  - User information
  - Review statistics
  - Detailed review analysis
- **Copy-Paste Friendly Output**: All tables are formatted for easy copying into:
  - Spreadsheets (Excel, Google Sheets)
  - Email clients
  - Text editors
  - Documentation tools

## Requirements

- Python 3.8+
- OpenAI API Key
- Google API Key

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd SentimentAnalyser
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
export OPENAI_API_KEY='your-openai-api-key'
export GOOGLE_API_KEY='your-google-api-key'
```

## Usage

1. Basic usage:
```python
from sentiment_analyzer import SentimentAnalyzer, ModelType

# Initialize analyzer
analyzer = SentimentAnalyzer()

# Analyze reviews
reviews = [
    {"text": "Great app!", "author": "User1", "stars": 5},
    {"text": "Could be better", "author": "User2", "stars": 3}
]

# Choose model(s)
model_type = ModelType.BOTH  # or ModelType.OPENAI or ModelType.GEMINI

# Run analysis
results = analyzer.analyze_reviews(reviews, model_type)
```

2. Output format:
- Console: Displays formatted tables with grid lines
- File: Generates `review_analysis.txt` with tab-separated tables for easy copying

## Output Sections

1. **Emotion Distribution**
   - Shows frequency of different emotions
   - Helps identify common sentiments

2. **User Information**
   - User details
   - Device information
   - Review previews
   - Emotional context

3. **Summary Statistics**
   - Total review count
   - Average ratings
   - Model distribution

4. **Detailed Review Analysis**
   - Comprehensive view of all reviews
   - Emotions and ratings
   - Device information

## File Output Format

The analysis is saved in `review_analysis.txt` with the following structure:
```
Analysis Time: [timestamp]

Emotion Distribution:
[tab-separated emotion data]

User Information:
[tab-separated user details]

Summary:
[statistics]

Detailed Review Analysis:
[tab-separated detailed review data]
```

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

[License Type] - See LICENSE file for details

## Documentation

For detailed code documentation and API references, see [DOCUMENTATION.md](DOCUMENTATION.md)

# Sentiment Analyzer Documentation

## Code Structure

### Main Components

1. **SentimentAnalyzer Class** (`sentiment_analyzer.py`)
   - Core class handling all sentiment analysis operations
   - Manages multiple model interactions
   - Processes and formats results

2. **ModelType Enum**
   - `OPENAI`: Use OpenAI GPT-3.5 model
   - `GEMINI`: Use Google Gemini model
   - `BOTH`: Use both models for analysis

## API Reference

### SentimentAnalyzer Class

#### Initialization
```python
analyzer = SentimentAnalyzer()
```

#### Methods

1. **analyze_reviews**
```python
def analyze_reviews(self, reviews: List[Dict], model_type: ModelType = ModelType.BOTH) -> pd.DataFrame
```
- Analyzes a list of reviews using specified model(s)
- Parameters:
  - `reviews`: List of dictionaries containing review data
  - `model_type`: Type of model to use (OPENAI/GEMINI/BOTH)
- Returns: DataFrame with analysis results

2. **extract_emotion**
```python
def extract_emotion(self, analysis: str) -> str
```
- Extracts emotion from analysis text
- Parameters:
  - `analysis`: Analysis text from model
- Returns: Extracted emotion as string

3. **format_table_dual**
```python
def format_table_dual(self, data: List[List], headers: List[str]) -> Dict[str, str]
```
- Formats data into display and copy-paste versions
- Parameters:
  - `data`: List of rows
  - `headers`: Column headers
- Returns: Dictionary with 'display' and 'copy' versions

4. **display_analysis**
```python
def display_analysis(self, df: pd.DataFrame) -> None
```
- Displays analysis results in console and saves to file
- Parameters:
  - `df`: DataFrame with analysis results
- Output sections:
  - Emotion Distribution
  - User Information
  - Summary Statistics
  - Detailed Review Analysis

### Data Structures

#### Review Input Format
```python
review = {
    "text": str,        # Review text
    "author": str,      # Author name
    "stars": int,       # Rating (1-5)
    "device": str,      # Optional: Device used
    "client_id": str,   # Optional: Client identifier
    "manufacturer": str # Optional: Device manufacturer
}
```

#### Analysis Output Format
The analysis results are saved in `review_analysis.txt` with tab-separated tables:

1. **Emotion Distribution Table**
```
Emotion  Count
Happy    5
Satisfied 3
```

2. **User Information Table**
```
Name  Device  Client ID  Emotion  Review Preview
User1 iPhone  12345     Happy    Great app...
```

3. **Detailed Review Analysis Table**
```
User  Emotion  Rating  Device  Review
User1 Happy    5 stars iPhone  Full review text...
```

## Output Customization

### Console Output
- Uses grid formatting with Unicode box-drawing characters
- Includes color coding using `colorama`
- Shows all analysis sections with decorative headers

### File Output
- Uses tab-separated format for easy copying
- Simple section headers without special characters
- Optimized for pasting into other applications

## Error Handling

1. **API Errors**
   - Handles OpenAI API errors
   - Manages Google API authentication issues
   - Provides meaningful error messages

2. **Data Validation**
   - Validates review input format
   - Checks for required fields
   - Handles missing optional fields

## Best Practices

1. **API Key Management**
   - Use environment variables for API keys
   - Never hardcode sensitive information
   - Implement proper error handling for missing keys

2. **Performance Optimization**
   - Batch processing for multiple reviews
   - Efficient memory usage in table formatting
   - Optimized file I/O operations

3. **Output Formatting**
   - Consistent table structures
   - Clear section separation
   - Copy-paste friendly formats

## Examples

### Basic Usage
```python
from sentiment_analyzer import SentimentAnalyzer, ModelType

# Initialize
analyzer = SentimentAnalyzer()

# Single review analysis
review = {
    "text": "Great app, really helpful!",
    "author": "User1",
    "stars": 5,
    "device": "iPhone"
}

# Analyze with both models
results = analyzer.analyze_reviews([review], ModelType.BOTH)

# Display results
analyzer.display_analysis(results)
```

### Batch Processing
```python
# Multiple reviews
reviews = [
    {
        "text": "Amazing features",
        "author": "User1",
        "stars": 5
    },
    {
        "text": "Needs improvement",
        "author": "User2",
        "stars": 3
    }
]

# Analyze with specific model
results = analyzer.analyze_reviews(reviews, ModelType.OPENAI)
```

## Troubleshooting

### Common Issues

1. **API Authentication**
   - Check environment variables
   - Verify API key validity
   - Ensure proper permissions

2. **Output Formatting**
   - Check terminal UTF-8 support
   - Verify file encoding
   - Ensure proper line endings

3. **Performance**
   - Monitor API rate limits
   - Check memory usage
   - Optimize batch sizes

## Future Enhancements

1. **Model Support**
   - Add support for more LLMs
   - Implement custom model options
   - Enhanced model comparison

2. **Output Formats**
   - Additional export formats (CSV, JSON)
   - Custom templating options
   - Interactive visualizations

3. **Analysis Features**
   - Sentiment trend analysis
   - Advanced emotion categorization
   - Automated insights generation

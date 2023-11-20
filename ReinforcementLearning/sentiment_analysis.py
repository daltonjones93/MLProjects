from transformers import pipeline

def analyze_sentiment(text):
    # Load the pre-trained sentiment analysis model
    sentiment_pipeline = pipeline("sentiment-analysis")

    # Perform sentiment analysis on the input text
    result = sentiment_pipeline(text)

    # Extract sentiment and confidence score
    sentiment = result[0]['label']
    confidence = result[0]['score']

    return sentiment, confidence

# Example usage
text_input = "username and password"
sentiment, confidence = analyze_sentiment(text_input)

print(f"Sentiment: {sentiment}, Confidence: {confidence:.4f}")
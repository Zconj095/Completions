import nltk
from nltk.tokenize import TweetTokenizer, word_tokenize
import time
import spacy
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize models
nlp = spacy.load("en_core_web_sm")
tweet_tokenizer = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=True)


def read_text_file(filename):
    """Read text file with error handling."""
    try:
        return Path(filename).read_text(encoding="utf-8")
    except FileNotFoundError:
        logger.error(f"File not found: {filename}")
        return ""
    except Exception as e:
        logger.error(f"Error reading file: {e}")
        return ""


def preprocess_text(text):
    """Clean and preprocess text."""
    if not text:
        return ""
    text = text.replace("\n", " ").replace("\r", " ")
    text = " ".join(text.split())  # Remove extra whitespace
    return text


def tokenize_nltk(text):
    """Tokenize using NLTK word tokenizer."""
    return word_tokenize(text)


def tokenize_spacy(text):
    """Tokenize using spaCy with additional token information."""
    doc = nlp(text)
    return [token.text for token in doc if not token.is_space]


def tokenize_tweet(text):
    """Tokenize tweets using NLTK's TweetTokenizer."""
    return tweet_tokenizer.tokenize(text)


def tokenize(text, method="spacy"):
    """Main tokenization function with method selection."""
    tokenizers = {
        "nltk": tokenize_nltk,
        "spacy": tokenize_spacy,
        "tweet": tokenize_tweet
    }
    
    if method not in tokenizers:
        logger.warning(f"Unknown method '{method}', using spacy")
        method = "spacy"
    
    return tokenizers[method](text)


def compare_tokenizers(text):
    """Compare different tokenization methods."""
    methods = ["nltk", "spacy", "tweet"]
    results = {}
    
    for method in methods:
        start_time = time.time()
        tokens = tokenize(text, method)
        duration = time.time() - start_time
        results[method] = {
            "tokens": tokens,
            "count": len(tokens),
            "time": duration
        }
    
    return results


def main():
    # Process regular text file
    filename = "__FolderPath__/__FileName__.__Extension__"
    text = read_text_file(filename)
    
    if text:
        text = preprocess_text(text)
        words = tokenize(text)
        logger.info(f"Tokenized {len(words)} words from file")
        print(f"Sample tokens: {words[:10]}...")
    
    # Process tweet
    tweet = "@EmpireStateBldg Central Park Tower is reaaaaally hiiiiiiigh"
    tweet_tokens = tokenize(tweet, method="tweet")
    logger.info(f"Tweet tokens: {tweet_tokens}")
    
    # Compare tokenizers on tweet
    print("\nTokenizer comparison:")
    comparison = compare_tokenizers(tweet)
    for method, result in comparison.items():
        print(f"{method}: {result['count']} tokens in {result['time']:.4f}s")
        print(f"  Tokens: {result['tokens']}")


if __name__ == "__main__":
    start = time.time()
    main()
    total_time = time.time() - start
    logger.info(f"Total execution time: {total_time:.4f}s")

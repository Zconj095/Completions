import nltk
from nltk.tokenize import TweetTokenizer, word_tokenize
import time
import spacy
from pathlib import Path
from typing import List, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Initialize models
nlp = spacy.load("en_core_web_sm")
tweet_tokenizer = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=True)


class TextTokenizer:
    """Enhanced text tokenization class with multiple tokenization methods."""
    
    def __init__(self):
        self.nlp = nlp
        self.tweet_tokenizer = tweet_tokenizer
    
    def read_text_file(self, filename: str) -> str:
        """Read text file with proper error handling."""
        try:
            file_path = Path(filename)
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {filename}")
            
            with open(file_path, "r", encoding="utf-8") as file:
                return file.read()
        except Exception as e:
            logger.error(f"Error reading file {filename}: {e}")
            return ""
    
    def preprocess_text(self, text: str) -> str:
        """Enhanced text preprocessing."""
        if not text:
            return ""
        
        # Replace newlines and multiple spaces
        text = text.replace("\n", " ").replace("\r", " ")
        # Remove multiple consecutive spaces
        text = " ".join(text.split())
        return text.strip()
    
    def tokenize_nltk(self, text: str) -> List[str]:
        """NLTK word tokenization."""
        return word_tokenize(text)
    
    def tokenize_spacy(self, text: str) -> List[str]:
        """SpaCy tokenization."""
        doc = self.nlp(text)
        return [token.text for token in doc]
    
    def tokenize_tweet(self, text: str) -> List[str]:
        """Tweet-specific tokenization using TweetTokenizer."""
        return self.tweet_tokenizer.tokenize(text)
    
    def tokenize_casual(self, text: str, preserve_case: bool = True, 
                       reduce_len: bool = True, strip_handles: bool = True) -> List[str]:
        """Casual tokenization for social media text."""
        return nltk.tokenize.casual.casual_tokenize(
            text, preserve_case=preserve_case, reduce_len=reduce_len, 
            strip_handles=strip_handles
        )
    
    def tokenize(self, text: str, method: str = "spacy") -> List[str]:
        """Main tokenization method with selectable tokenizer."""
        methods = {
            "nltk": self.tokenize_nltk,
            "spacy": self.tokenize_spacy,
            "tweet": self.tokenize_tweet,
            "casual": self.tokenize_casual
        }
        
        if method not in methods:
            logger.warning(f"Unknown method '{method}', using spacy")
            method = "spacy"
        
        return methods[method](text)


def benchmark_tokenizers(text: str, tokenizer: TextTokenizer) -> None:
    """Benchmark different tokenization methods."""
    methods = ["nltk", "spacy", "tweet", "casual"]
    
    print("\nTokenization Benchmark:")
    print("-" * 50)
    
    for method in methods:
        start_time = time.time()
        tokens = tokenizer.tokenize(text, method)
        end_time = time.time()
        
        print(f"{method.upper():8}: {len(tokens):6} tokens in {end_time - start_time:.4f}s")


def main():
    """Main function with enhanced functionality."""
    tokenizer = TextTokenizer()
    
    # Example with file reading (update path as needed)
    filename = "sample_text.txt"  # Replace with actual file path
    
    try:
        # Read and process file
        text = tokenizer.read_text_file(filename)
        if text:
            processed_text = tokenizer.preprocess_text(text)
            words = tokenizer.tokenize(processed_text, method="spacy")
            print(f"File tokens (first 10): {words[:10]}")
            benchmark_tokenizers(processed_text[:1000], tokenizer)  # Use first 1000 chars
    except Exception as e:
        logger.warning(f"Skipping file processing: {e}")
    
    # Social media text examples
    social_media_texts = [
        "@EmpireStateBldg Central Park Tower is reaaaaally hiiiiiiigh",
        "OMG this is sooooo cool!!! #amazing #wow 😍",
        "Check out https://example.com @user123 #hashtag",
        "LOL that's hilarious 😂😂😂 RT @someone: great tweet!"
    ]
    
    print("\nSocial Media Text Tokenization:")
    print("-" * 50)
    
    for i, tweet in enumerate(social_media_texts, 1):
        print(f"\nExample {i}: {tweet}")
        
        # Compare different tokenization methods
        for method in ["tweet", "casual", "spacy"]:
            tokens = tokenizer.tokenize(tweet, method)
            print(f"  {method:6}: {tokens}")


if __name__ == "__main__":
    start = time.time()
    main()
    print(f"\nTotal execution time: {time.time() - start:.4f}s")

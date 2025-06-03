import csv
import nltk
import string
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Union
from nltk.probability import FreqDist
from nltk.corpus import stopwords

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextProcessor:
    def __init__(self, language: str = 'english'):
        """Initialize the text processor with specified language."""
        self.language = language
        self._download_nltk_data()
        
    def _download_nltk_data(self):
        """Download required NLTK data if not present."""
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('punkt')
            nltk.download('stopwords')
    
    def read_csv(self, csv_file: Union[str, Path], column: int = 0) -> List[str]:
        """Read data from CSV file with error handling."""
        try:
            with open(csv_file, 'r', encoding='utf-8') as fp:
                reader = csv.reader(fp, delimiter=',', quotechar='"')
                data = [row[column] for row in reader if len(row) > column]
            logger.info(f"Successfully read {len(data)} rows from {csv_file}")
            return data
        except FileNotFoundError:
            logger.error(f"File {csv_file} not found")
            return []
        except Exception as e:
            logger.error(f"Error reading CSV: {e}")
            return []

    def read_text_file(self, filename: Union[str, Path]) -> str:
        """Read text file with error handling."""
        try:
            with open(filename, "r", encoding="utf-8") as file:
                content = file.read()
            logger.info(f"Successfully read {filename}")
            return content
        except FileNotFoundError:
            logger.error(f"File {filename} not found")
            return ""
        except Exception as e:
            logger.error(f"Error reading file: {e}")
            return ""

    def preprocess_text(self, text: str, remove_punctuation: bool = True, 
                       remove_numbers: bool = False) -> str:
        """Enhanced text preprocessing with options."""
        if not text:
            return ""
        
        # Replace newlines and multiple spaces
        text = text.replace("\n", " ").replace("\r", " ")
        text = " ".join(text.split())  # Remove multiple spaces
        
        if remove_punctuation:
            text = text.translate(str.maketrans('', '', string.punctuation))
        
        if remove_numbers:
            text = ''.join(char for char in text if not char.isdigit())
        
        return text.strip()

    def tokenize(self, text: str, min_length: int = 1) -> List[str]:
        """Tokenize text with minimum length filter."""
        if not text:
            return []
        
        tokens = nltk.word_tokenize(text)
        return [token for token in tokens if len(token) >= min_length]

    def create_frequency_dist(self, words: List[str]) -> FreqDist:
        """Create frequency distribution from words."""
        return FreqDist(word.lower() for word in words if word.isalpha())

    def get_stopwords(self, custom_stopwords: Optional[List[str]] = None) -> List[str]:
        """Get combined stopwords from NLTK and custom list."""
        try:
            nltk_stopwords = set(stopwords.words(self.language))
        except OSError:
            logger.warning(f"Stopwords for {self.language} not found, using empty set")
            nltk_stopwords = set()
        
        if custom_stopwords:
            nltk_stopwords.update(custom_stopwords)
        
        return list(nltk_stopwords)

    def compile_stopwords_by_frequency(self, text: str, 
                                     frequency_threshold: Optional[int] = None,
                                     percentage_threshold: float = 0.02) -> List[str]:
        """Generate stopwords list based on frequency analysis."""
        if not text:
            return []
        
        processed_text = self.preprocess_text(text)
        words = self.tokenize(processed_text, min_length=2)
        
        if not words:
            return []
        
        freq_dist = self.create_frequency_dist(words)
        words_with_frequencies = [(word, freq) for word, freq in freq_dist.items()]
        sorted_words = sorted(words_with_frequencies, key=lambda x: x[1], reverse=True)
        
        if frequency_threshold:
            stopwords_list = [word for word, freq in sorted_words if freq >= frequency_threshold]
        else:
            cutoff_index = int(percentage_threshold * len(sorted_words))
            stopwords_list = [word for word, _ in sorted_words[:cutoff_index]]
        
        logger.info(f"Generated {len(stopwords_list)} stopwords from frequency analysis")
        return stopwords_list

    def remove_stopwords(self, text: str, custom_stopwords: Optional[List[str]] = None,
                        min_word_length: int = 2) -> List[str]:
        """Remove stopwords from text and return filtered words."""
        stopwords_list = self.get_stopwords(custom_stopwords)
        processed_text = self.preprocess_text(text)
        words = self.tokenize(processed_text, min_length=min_word_length)
        
        filtered_words = [
            word for word in words 
            if word.lower() not in stopwords_list and word.isalpha()
        ]
        
        logger.info(f"Filtered from {len(words)} to {len(filtered_words)} words")
        return filtered_words

    def get_word_frequencies(self, words: List[str], top_n: int = 50) -> List[Tuple[str, int]]:
        """Get top N most frequent words."""
        freq_dist = self.create_frequency_dist(words)
        return freq_dist.most_common(top_n)

def main():
    """Enhanced main function with better organization."""
    processor = TextProcessor()
    
    # Configuration
    text_file = "Folder_Name/File_Name.txt"
    stopwords_csv = "stopwords.csv"
    
    # Read text
    text = processor.read_text_file(text_file)
    if not text:
        logger.error("No text content found")
        return
    
    # Method 1: Use NLTK stopwords
    print("=== Using NLTK Stopwords ===")
    filtered_words_nltk = processor.remove_stopwords(text)
    print(f"Filtered words (first 20): {filtered_words_nltk[:20]}")
    
    # Method 2: Use custom stopwords from CSV (if file exists)
    custom_stopwords = processor.read_csv(stopwords_csv)
    if custom_stopwords:
        print("\n=== Using Custom Stopwords ===")
        filtered_words_custom = processor.remove_stopwords(text, custom_stopwords)
        print(f"Filtered words (first 20): {filtered_words_custom[:20]}")
    
    # Method 3: Generate stopwords by frequency
    print("\n=== Using Frequency-based Stopwords ===")
    freq_stopwords = processor.compile_stopwords_by_frequency(
        text, frequency_threshold=100
    )
    print(f"Generated stopwords: {freq_stopwords[:20]}")
    
    # Get word frequencies
    print("\n=== Top 20 Most Frequent Words ===")
    top_words = processor.get_word_frequencies(filtered_words_nltk, top_n=20)
    for word, freq in top_words:
        print(f"{word}: {freq}")

if __name__ == '__main__':
    main()
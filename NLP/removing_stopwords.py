import csv
import nltk
from nltk.probability import FreqDist
from nltk.corpus import stopwords
import string
import logging
from pathlib import Path
from typing import List, Tuple, Optional

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextProcessor:
    def __init__(self):
        self._download_nltk_data()
        self.stopwords = set(stopwords.words('english'))
    
    def _download_nltk_data(self):
        """Download required NLTK data if not present."""
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('punkt')
            nltk.download('stopwords')
    
    def read_csv(self, csv_file: str) -> List[str]:
        """Read stopwords from CSV file."""
        try:
            with open(csv_file, 'r', encoding='utf-8') as fp:
                reader = csv.reader(fp, delimiter=',', quotechar='"')
                return [row[0] for row in reader if row]
        except FileNotFoundError:
            logger.error(f"CSV file {csv_file} not found")
            return []
    
    def read_text_file(self, filename: str) -> str:
        """Read text from file with error handling."""
        try:
            with open(filename, "r", encoding="utf-8") as file:
                return file.read()
        except FileNotFoundError:
            logger.error(f"Text file {filename} not found")
            return ""
    
    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess text."""
        # Replace newlines and normalize whitespace
        text = text.replace("\n", " ").replace("\r", " ")
        # Remove extra whitespace
        text = " ".join(text.split())
        return text
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text and filter out punctuation."""
        tokens = nltk.tokenize.word_tokenize(text)
        # Filter out punctuation and numbers
        return [token for token in tokens if token.isalpha()]
    
    def create_frequency_dist(self, words: List[str]) -> FreqDist:
        """Create frequency distribution from words."""
        return FreqDist(word.lower() for word in words)
    
    def compile_stopwords_by_frequency(self, text: str, 
                                     frequency_threshold: Optional[int] = None,
                                     percentage_cutoff: float = 0.02) -> List[str]:
        """Generate stopwords based on frequency analysis."""
        text = self.preprocess_text(text)
        words = self.tokenize(text)
        freq_dist = self.create_frequency_dist(words)
        
        words_with_frequencies = [(word, freq_dist[word]) for word in freq_dist.keys()]
        sorted_words = sorted(words_with_frequencies, key=lambda x: x[1], reverse=True)
        
        if frequency_threshold:
            # Use frequency cutoff
            stopwords = [word for word, freq in sorted_words if freq > frequency_threshold]
        else:
            # Use percentage cutoff
            cutoff_index = int(percentage_cutoff * len(sorted_words))
            stopwords = [word for word, _ in sorted_words[:cutoff_index]]
        
        return stopwords
    
    def remove_stopwords(self, text: str, custom_stopwords: Optional[List[str]] = None) -> List[str]:
        """Remove stopwords from text."""
        text = self.preprocess_text(text)
        words = self.tokenize(text)
        
        # Combine default and custom stopwords
        all_stopwords = self.stopwords.copy()
        if custom_stopwords:
            all_stopwords.update(word.lower() for word in custom_stopwords)
        
        return [word for word in words if word.lower() not in all_stopwords]
    
    def analyze_text(self, filename: str, 
                    custom_stopwords_file: Optional[str] = None,
                    frequency_threshold: Optional[int] = None) -> dict:
        """Comprehensive text analysis."""
        text = self.read_text_file(filename)
        if not text:
            return {}
        
        # Load custom stopwords if provided
        custom_stopwords = []
        if custom_stopwords_file:
            custom_stopwords = self.read_csv(custom_stopwords_file)
        
        # Process text
        filtered_words = self.remove_stopwords(text, custom_stopwords)
        freq_based_stopwords = self.compile_stopwords_by_frequency(text, frequency_threshold)
        
        # Create frequency distribution
        freq_dist = self.create_frequency_dist(filtered_words)
        
        return {
            'original_word_count': len(self.tokenize(text)),
            'filtered_word_count': len(filtered_words),
            'filtered_words': filtered_words,
            'frequency_based_stopwords': freq_based_stopwords,
            'most_common_words': freq_dist.most_common(20),
            'vocabulary_size': len(set(filtered_words))
        }

def main():
    processor = TextProcessor()
    
    # Example usage
    filename = "Folder_Name/File_Name.txt"
    
    # Check if file exists
    if not Path(filename).exists():
        logger.error(f"File {filename} does not exist")
        return
    
    # Perform analysis
    results = processor.analyze_text(
        filename=filename,
        custom_stopwords_file="stopwords.csv",  # Optional
        frequency_threshold=100  # Optional
    )
    
    if results:
        print(f"Original word count: {results['original_word_count']}")
        print(f"Filtered word count: {results['filtered_word_count']}")
        print(f"Vocabulary size: {results['vocabulary_size']}")
        print(f"Most common words: {results['most_common_words'][:10]}")
        print(f"Frequency-based stopwords: {results['frequency_based_stopwords'][:10]}")

if __name__ == '__main__':
    main()
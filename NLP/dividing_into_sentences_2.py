import time
import nltk
import spacy
import argparse
import logging
from pathlib import Path
from typing import List, Optional
import json
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SentenceSegmenter:
    """Enhanced sentence segmentation with multiple methods and preprocessing options."""
    
    def __init__(self):
        self.tokenizer = None
        self.nlp = None
        self._load_resources()
    
    def _load_resources(self):
        """Load NLTK and spaCy resources with error handling."""
        # Load NLTK tokenizer
        try:
            self.tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")
            logger.info("NLTK Punkt tokenizer loaded successfully")
        except LookupError:
            logger.info("Downloading NLTK punkt tokenizer...")
            nltk.download("punkt")
            self.tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")
        
        # Load spaCy model
        try:
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("spaCy model loaded successfully")
        except OSError:
            logger.warning("spaCy model 'en_core_web_sm' not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None

    def read_text_file(self, filename: str) -> str:
        """Read text file with improved encoding handling."""
        path = Path(filename)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {filename}")
        
        # Try different encodings
        encodings = ['utf-8', 'latin-1', 'cp1252']
        for encoding in encodings:
            try:
                with open(path, "r", encoding=encoding) as file:
                    content = file.read()
                    logger.info(f"File read successfully with {encoding} encoding")
                    return content
            except UnicodeDecodeError:
                continue
        
        raise ValueError(f"Could not decode file {filename} with any supported encoding")

    def preprocess_text(self, text: str, clean_whitespace: bool = True, 
                       remove_extra_newlines: bool = True, 
                       normalize_quotes: bool = True) -> str:
        """Enhanced text preprocessing with configurable options."""
        if clean_whitespace:
            text = ' '.join(text.split())
        
        if remove_extra_newlines:
            text = re.sub(r'\n+', ' ', text)
        
        if normalize_quotes:
            # Normalize double quotes
            text = text.replace('“', '"').replace('”', '"').replace('‟', '"')
            # Normalize single quotes
            text = text.replace("‘", "'").replace("’", "'").replace("‚", "'")
        
        return text.strip()

    def segment_nltk(self, text: str) -> List[str]:
        """Segment text using NLTK."""
        if self.tokenizer is None:
            raise RuntimeError("NLTK tokenizer not available")
        return [sent.strip() for sent in self.tokenizer.tokenize(text) if sent.strip()]

    def segment_spacy(self, text: str) -> List[str]:
        """Segment text using spaCy."""
        if self.nlp is None:
            raise RuntimeError("spaCy model not available")
        doc = self.nlp(text)
        return [sent.text.strip() for sent in doc.sents if sent.text.strip()]

    def segment_simple(self, text: str) -> List[str]:
        """Simple regex-based sentence segmentation as fallback."""
        # Basic sentence ending pattern
        pattern = r'(?<=[.!?])\s+'
        sentences = re.split(pattern, text)
        return [sent.strip() for sent in sentences if sent.strip()]

    def segment_text(self, text: str, method: str = "nltk") -> List[str]:
        """Main segmentation method with fallback options."""
        methods = {
            "nltk": self.segment_nltk,
            "spacy": self.segment_spacy,
            "simple": self.segment_simple
        }
        
        if method not in methods:
            raise ValueError(f"Unknown method: {method}. Choose from {list(methods.keys())}")
        
        try:
            return methods[method](text)
        except RuntimeError as e:
            logger.warning(f"{method} method failed: {e}")
            # Fallback to simple method
            logger.info("Falling back to simple regex method")
            return self.segment_simple(text)

    def filter_sentences(self, sentences: List[str], 
                        min_length: int = 3, 
                        max_length: int = 1000,
                        remove_numbers_only: bool = True) -> List[str]:
        """Filter sentences based on various criteria."""
        filtered = []
        for sent in sentences:
            # Length filter
            if not (min_length <= len(sent) <= max_length):
                continue
            
            # Remove sentences that are only numbers/punctuation
            if remove_numbers_only and re.match(r'^[\d\s\W]+$', sent):
                continue
            
            filtered.append(sent)
        
        return filtered

    def save_sentences(self, sentences: List[str], output_file: str, 
                      format_type: str = "txt") -> None:
        """Save sentences in different formats."""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format_type == "txt":
            with open(output_path, "w", encoding="utf-8") as f:
                for i, sent in enumerate(sentences, 1):
                    f.write(f"{i}: {sent}\n")
        
        elif format_type == "json":
            data = {
                "sentences": sentences,
                "count": len(sentences),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        
        elif format_type == "csv":
            import csv
            with open(output_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["Index", "Sentence", "Length"])
                for i, sent in enumerate(sentences, 1):
                    writer.writerow([i, sent, len(sent)])
        
        logger.info(f"Sentences saved to {output_path} in {format_type} format")

def main():
    parser = argparse.ArgumentParser(
        description="Enhanced sentence segmentation tool with multiple methods and options.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python script.py input.txt
  python script.py input.txt -m spacy -o output.json --format json
  python script.py input.txt --min-length 10 --max-length 500
        """
    )
    
    parser.add_argument("input_file", help="Path to the input text file")
    parser.add_argument("-m", "--method", choices=["nltk", "spacy", "simple"], 
                       default="nltk", help="Sentence segmentation method")
    parser.add_argument("-o", "--output", help="Output file path")
    parser.add_argument("--format", choices=["txt", "json", "csv"], 
                       default="txt", help="Output format")
    parser.add_argument("--min-length", type=int, default=3, 
                       help="Minimum sentence length")
    parser.add_argument("--max-length", type=int, default=1000, 
                       help="Maximum sentence length")
    parser.add_argument("--no-filter", action="store_true", 
                       help="Skip sentence filtering")
    parser.add_argument("--no-preprocess", action="store_true", 
                       help="Skip text preprocessing")
    parser.add_argument("-v", "--verbose", action="store_true", 
                       help="Enable verbose logging")
    
    try:
        args = parser.parse_args()
    except SystemExit:
        return 1
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        segmenter = SentenceSegmenter()
        
        # Read and process text
        text = segmenter.read_text_file(args.input_file)
        logger.info(f"Read {len(text)} characters from {args.input_file}")
        
        if not args.no_preprocess:
            text = segmenter.preprocess_text(text)
            logger.info("Text preprocessing completed")
        
        # Segment text
        sentences = segmenter.segment_text(text, method=args.method)
        logger.info(f"Initial segmentation: {len(sentences)} sentences")
        
        # Filter sentences
        if not args.no_filter:
            sentences = segmenter.filter_sentences(
                sentences, 
                min_length=args.min_length,
                max_length=args.max_length
            )
            logger.info(f"After filtering: {len(sentences)} sentences")
        
        # Display results
        print(f"\nFound {len(sentences)} sentences:\n")
        for i, sent in enumerate(sentences[:10], 1):  # Show first 10
            print(f"{i}: {sent}")
        
        if len(sentences) > 10:
            print(f"\n... and {len(sentences) - 10} more sentences")
        
        # Save if requested
        if args.output:
            segmenter.save_sentences(sentences, args.output, args.format)
        
        # Statistics
        if sentences:
            avg_length = sum(len(s) for s in sentences) / len(sentences)
            print(f"\nStatistics:")
            print(f"Total sentences: {len(sentences)}")
            print(f"Average length: {avg_length:.1f} characters")
            print(f"Shortest: {min(len(s) for s in sentences)} characters")
            print(f"Longest: {max(len(s) for s in sentences)} characters")
    
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1
    
    return 0

if __name__ == '__main__':
    start_time = time.time()
    exit_code = main()
    print(f"\nExecution time: {time.time() - start_time:.2f} seconds")
    # Don't call exit() to avoid SystemExit exception in interactive environments
    if exit_code != 0:
        print(f"Program exited with code: {exit_code}")

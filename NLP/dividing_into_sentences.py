import time
import logging
from pathlib import Path
from typing import List, Optional, Union
import nltk
import spacy

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SentenceDivider:
    def __init__(self, method: str = "nltk", spacy_model: str = "en_core_web_sm"):
        self.method = method.lower()
        self.tokenizer = None
        self.nlp = None
        
        if self.method in ["nltk", "both"]:
            self._setup_nltk()
        
        if self.method in ["spacy", "both"]:
            self._setup_spacy(spacy_model)
    
    def _setup_nltk(self):
        try:
            self.tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")
        except LookupError:
            logger.warning("NLTK punkt tokenizer not found. Downloading...")
            nltk.download('punkt')
            self.tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")
    
    def _setup_spacy(self, model: str):
        try:
            self.nlp = spacy.load(model)
        except OSError:
            logger.error(f"spaCy model '{model}' not found. Please install it with: python -m spacy download {model}")
            raise
    
    def read_text_file(self, filename: Union[str, Path]) -> str:
        """Read text file with error handling."""
        try:
            file_path = Path(filename)
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {filename}")
            
            with open(file_path, "r", encoding="utf-8") as file:
                return file.read()
        except Exception as e:
            logger.error(f"Error reading file {filename}: {e}")
            raise
    
    def preprocess_text(self, text: str, remove_empty: bool = True) -> str:
        """Enhanced text preprocessing."""
        if not text:
            return ""
        
        # Replace multiple whitespace characters with single space
        text = " ".join(text.split())
        
        if remove_empty:
            # Remove empty lines and excessive whitespace
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            text = " ".join(lines)
        
        return text
    
    def divide_into_sentences_nltk(self, text: str) -> List[str]:
        """Divide text into sentences using NLTK."""
        if not self.tokenizer:
            raise RuntimeError("NLTK tokenizer not initialized")
        
        sentences = self.tokenizer.tokenize(text)
        return [sentence.strip() for sentence in sentences if sentence.strip()]
    
    def divide_into_sentences_spacy(self, text: str) -> List[str]:
        """Divide text into sentences using spaCy."""
        if not self.nlp:
            raise RuntimeError("spaCy model not initialized")
        
        doc = self.nlp(text)
        return [sentence.text.strip() for sentence in doc.sents if sentence.text.strip()]
    
    def divide_into_sentences(self, text: str) -> Union[List[str], dict]:
        """Divide text into sentences using the specified method."""
        if not text:
            return []
        
        if self.method == "nltk":
            return self.divide_into_sentences_nltk(text)
        elif self.method == "spacy":
            return self.divide_into_sentences_spacy(text)
        elif self.method == "both":
            return {
                "nltk": self.divide_into_sentences_nltk(text),
                "spacy": self.divide_into_sentences_spacy(text)
            }
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def process_file(self, filename: Union[str, Path], 
                    output_file: Optional[Union[str, Path]] = None) -> Union[List[str], dict]:
        """Process a file and optionally save results."""
        logger.info(f"Processing file: {filename}")
        
        text = self.read_text_file(filename)
        text = self.preprocess_text(text)
        sentences = self.divide_into_sentences(text)
        
        logger.info(f"Found {len(sentences) if isinstance(sentences, list) else len(sentences['nltk'])} sentences")
        
        if output_file:
            self._save_sentences(sentences, output_file)
        
        return sentences
    
    def _save_sentences(self, sentences: Union[List[str], dict], 
                       output_file: Union[str, Path]):
        """Save sentences to a file."""
        output_path = Path(output_file)
        
        with open(output_path, "w", encoding="utf-8") as f:
            if isinstance(sentences, list):
                for i, sentence in enumerate(sentences, 1):
                    f.write(f"{i}. {sentence}\n\n")
            else:
                for method, sent_list in sentences.items():
                    f.write(f"=== {method.upper()} ===\n")
                    for i, sentence in enumerate(sent_list, 1):
                        f.write(f"{i}. {sentence}\n")
                    f.write("\n")
        
        logger.info(f"Results saved to: {output_file}")

def main():
    # Example usage
    divider = SentenceDivider(method="nltk")  # or "spacy" or "both"
    
    try:
        sentences = divider.process_file(
            "Folder_Name/File_Name.txt",
            output_file="output_sentences.txt"  # Optional
        )
        
        # Print first 3 sentences as preview
        if isinstance(sentences, list):
            for i, sentence in enumerate(sentences[:3], 1):
                print(f"{i}. {sentence}")
        else:
            print("NLTK results (first 3):")
            for i, sentence in enumerate(sentences["nltk"][:3], 1):
                print(f"{i}. {sentence}")
    
    except Exception as e:
        logger.error(f"Error in main: {e}")

if __name__ == '__main__':
    start = time.time()
    main()
    print(f"Execution time: {time.time() - start:.2f} seconds")

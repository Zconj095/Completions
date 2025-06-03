from nltk.stem.snowball import SnowballStemmer
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer
import nltk

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class StemmerManager:
    def __init__(self, language='english', algorithm='snowball'):
        self.language = language
        self.algorithm = algorithm
        self.stemmer = self._get_stemmer()
    
    def _get_stemmer(self):
        if self.algorithm == 'snowball':
            return SnowballStemmer(self.language)
        elif self.algorithm == 'porter':
            return PorterStemmer()
        elif self.algorithm == 'lancaster':
            return LancasterStemmer()
        else:
            raise ValueError("Unsupported algorithm. Use 'snowball', 'porter', or 'lancaster'")
    
    def stem_word(self, word):
        """Stem a single word"""
        return self.stemmer.stem(word.lower())
    
    def stem_words(self, words):
        """Stem a list of words"""
        return [self.stem_word(word) for word in words]
    
    def stem_sentence(self, sentence):
        """Stem all words in a sentence"""
        words = nltk.word_tokenize(sentence)
        stemmed = self.stem_words(words)
        return ' '.join(stemmed)
    
    def compare_algorithms(self, words):
        """Compare stemming results across different algorithms"""
        algorithms = ['snowball', 'porter', 'lancaster']
        results = {}
        
        for algo in algorithms:
            try:
                temp_stemmer = StemmerManager(self.language, algo)
                results[algo] = temp_stemmer.stem_words(words)
            except ValueError:
                results[algo] = "Not supported for this language"
        
        return results

def main():
    # Test words
    english_words = ['leaf', 'leaves', 'booking', 'writing', 'completed', 
                    'stemming', 'skiing', 'skies', 'running', 'ran', 'runs']
    spanish_words = ['caminando', 'amigo', 'bueno', 'corriendo', 'libros']
    
    # English stemming
    print("=== English Stemming ===")
    english_stemmer = StemmerManager('english', 'snowball')
    
    print(f"Original: {english_words}")
    print(f"Stemmed:  {english_stemmer.stem_words(english_words)}")
    
    # Sentence stemming
    sentence = "The leaves are falling from the trees while children are running"
    print(f"\nOriginal sentence: {sentence}")
    print(f"Stemmed sentence:  {english_stemmer.stem_sentence(sentence)}")
    
    # Algorithm comparison
    print("\n=== Algorithm Comparison ===")
    comparison = english_stemmer.compare_algorithms(english_words[:5])
    for algo, results in comparison.items():
        print(f"{algo.capitalize()}: {results}")
    
    # Spanish stemming
    print("\n=== Spanish Stemming ===")
    spanish_stemmer = StemmerManager('spanish', 'snowball')
    print(f"Original: {spanish_words}")
    print(f"Stemmed:  {spanish_stemmer.stem_words(spanish_words)}")

if __name__ == '__main__':
    main()
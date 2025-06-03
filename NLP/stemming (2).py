from nltk.stem.snowball import SnowballStemmer
from nltk.stem import PorterStemmer
import nltk

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class StemmerAnalyzer:
    def __init__(self, language='english'):
        self.snowball_stemmer = SnowballStemmer(language)
        self.porter_stemmer = PorterStemmer()
        self.language = language
    
    def stem_word(self, word, stemmer_type='snowball'):
        """Stem a single word using specified stemmer."""
        if stemmer_type == 'snowball':
            return self.snowball_stemmer.stem(word.lower())
        elif stemmer_type == 'porter':
            return self.porter_stemmer.stem(word.lower())
        else:
            raise ValueError("stemmer_type must be 'snowball' or 'porter'")
    
    def stem_text(self, text, stemmer_type='snowball'):
        """Stem all words in a text string."""
        words = nltk.word_tokenize(text)
        return [self.stem_word(word, stemmer_type) for word in words if word.isalpha()]
    
    def compare_stemmers(self, words):
        """Compare results from different stemmers."""
        results = []
        for word in words:
            snowball_stem = self.stem_word(word, 'snowball')
            porter_stem = self.stem_word(word, 'porter')
            results.append({
                'original': word,
                'snowball': snowball_stem,
                'porter': porter_stem,
                'same': snowball_stem == porter_stem
            })
        return results

def main():
    # Initialize analyzer
    analyzer = StemmerAnalyzer('english')
    
    # Test words
    words = ['leaf', 'leaves', 'booking', 'writing', 'completed', 'stemming', 
             'skiing', 'skies', 'running', 'runner', 'easily', 'fairly']
    
    print("=== Word Stemming Results ===")
    stemmed_words = [analyzer.stem_word(word) for word in words]
    for original, stemmed in zip(words, stemmed_words):
        print(f"{original:12} -> {stemmed}")
    
    print("\n=== Stemmer Comparison ===")
    comparison = analyzer.compare_stemmers(words[:6])
    for result in comparison:
        print(f"{result['original']:12} | Snowball: {result['snowball']:8} | Porter: {result['porter']:8} | Same: {result['same']}")
    
    print("\n=== Text Stemming ===")
    sample_text = "The runners were running quickly through the beautiful gardens with leaves falling."
    stemmed_text = analyzer.stem_text(sample_text)
    print(f"Original: {sample_text}")
    print(f"Stemmed:  {' '.join(stemmed_text)}")

if __name__ == '__main__':
    main()
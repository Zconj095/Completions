import spacy
from typing import List, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ClauseSplitter:
    """A class for splitting sentences into grammatical clauses using spaCy."""
    
    def __init__(self, model_name: str = 'en_core_web_sm'):
        """Initialize the clause splitter with a spaCy model."""
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            logger.error(f"Model '{model_name}' not found. Install with: python -m spacy download {model_name}")
            raise
    
    def find_root_token(self, doc: spacy.tokens.Doc) -> Optional[spacy.tokens.Token]:
        """Find the root token of the sentence."""
        for token in doc:
            if token.dep_ == "ROOT":
                return token
        return None
    
    def find_subordinate_verbs(self, doc: spacy.tokens.Doc, root_token: spacy.tokens.Token) -> List[spacy.tokens.Token]:
        """Find verbs that are direct children of the root token."""
        subordinate_verbs = []
        for token in doc:
            if (token.pos_ == "VERB" and 
                token != root_token and
                len(list(token.ancestors)) == 1 and 
                list(token.ancestors)[0] == root_token):
                subordinate_verbs.append(token)
        return subordinate_verbs
    
    def find_all_clause_verbs(self, doc: spacy.tokens.Doc) -> List[spacy.tokens.Token]:
        """Find all verbs that can anchor clauses."""
        clause_verbs = []
        for token in doc:
            if token.pos_ == "VERB" and token.dep_ in ["ROOT", "ccomp", "xcomp", "advcl", "relcl"]:
                clause_verbs.append(token)
        return clause_verbs
    
    def get_clause_span(self, verb: spacy.tokens.Token, doc: spacy.tokens.Doc, 
                       all_verbs: List[spacy.tokens.Token]) -> Tuple[int, int]:
        """Get the token span for a clause anchored by a verb."""
        # Start with the verb itself
        min_idx = verb.i
        max_idx = verb.i
        
        # Expand to include all dependents
        def expand_span(token: spacy.tokens.Token):
            nonlocal min_idx, max_idx
            min_idx = min(min_idx, token.i)
            max_idx = max(max_idx, token.i)
            
            for child in token.children:
                # Don't include other main verbs in this clause
                if child not in all_verbs or child.dep_ in ["aux", "auxpass"]:
                    expand_span(child)
        
        expand_span(verb)
        return (min_idx, max_idx + 1)  # +1 for exclusive end index
    
    def has_subject(self, clause: spacy.tokens.Span) -> bool:
        """Check if a clause contains a subject."""
        return any(token.dep_ in ["nsubj", "nsubjpass", "csubj"] for token in clause)
    
    def print_token_analysis(self, doc: spacy.tokens.Doc) -> None:
        """Print detailed token analysis for debugging."""
        print("\nToken Analysis:")
        print("-" * 80)
        print(f"{'Token':<12} {'Index':<6} {'POS':<8} {'Dep':<12} {'Ancestors':<20} {'Children'}")
        print("-" * 80)
        
        for token in doc:
            ancestors = [t.text for t in token.ancestors]
            children = [t.text for t in token.children]
            print(f"{token.text:<12} {token.i:<6} {token.pos_:<8} {token.dep_:<12} "
                  f"{str(ancestors):<20} {children}")
    
    def split_into_clauses(self, sentence: str, verbose: bool = False) -> List[str]:
        """
        Split a sentence into clauses.
        
        Args:
            sentence: The input sentence to split
            verbose: Whether to print debugging information
            
        Returns:
            List of clause texts
        """
        if not sentence.strip():
            return []
        
        doc = self.nlp(sentence)
        
        if verbose:
            self.print_token_analysis(doc)
            print(f"\nOriginal sentence: {sentence}")
        
        # Find all verbs that can anchor clauses
        clause_verbs = self.find_all_clause_verbs(doc)
        
        if not clause_verbs:
            logger.warning("No clause verbs found in sentence")
            return [sentence]
        
        # Get spans for each clause
        clause_spans = []
        for verb in clause_verbs:
            start, end = self.get_clause_span(verb, doc, clause_verbs)
            if start < end:
                clause_span = doc[start:end]
                # Only include clauses with subjects or that are very short
                if self.has_subject(clause_span) or len(clause_span) <= 3:
                    clause_spans.append(clause_span)
        
        # Sort clauses by their position in the sentence
        clause_spans.sort(key=lambda span: span.start)
        
        # Remove overlapping clauses (keep the longer one)
        filtered_spans = []
        for span in clause_spans:
            overlap = False
            for existing in filtered_spans:
                if (span.start < existing.end and span.end > existing.start):
                    # If there's overlap, keep the longer span
                    if len(span) > len(existing):
                        filtered_spans.remove(existing)
                    else:
                        overlap = True
                    break
            if not overlap:
                filtered_spans.append(span)
        
        clause_texts = [span.text.strip() for span in filtered_spans if span.text.strip()]
        
        if verbose:
            print(f"Found clauses: {clause_texts}")
        
        return clause_texts

def main():
    """Test the clause splitter with various sentence types."""
    splitter = ClauseSplitter()
    
    test_sentences = [
        "He eats cheese, but he won't eat ice cream.",
        "If it rains later, we won't be able to go to the park.",
        "The book that I bought yesterday is very interesting.",
        "Although she was tired, she continued working on her project.",
        "I think that you should visit the museum when you have time."
    ]
    
    for sentence in test_sentences:
        print(f"\nSentence: {sentence}")
        clauses = splitter.split_into_clauses(sentence, verbose=True)
        print(f"Clauses: {clauses}")
        print("-" * 60)

if __name__ == "__main__":
    main()

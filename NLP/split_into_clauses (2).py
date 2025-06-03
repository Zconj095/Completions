import spacy
from typing import List, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ClauseSplitter:
    """A class for splitting sentences into grammatical clauses using spaCy NLP."""
    
    def __init__(self, model_name: str = 'en_core_web_sm'):
        """Initialize the clause splitter with a spaCy model."""
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            logger.error(f"Failed to load spaCy model '{model_name}'. Please install it with: python -m spacy download {model_name}")
            raise
    
    def find_root_token(self, doc) -> Optional[spacy.tokens.Token]:
        """Find the root token of the sentence."""
        for token in doc:
            if token.dep_ == "ROOT":
                return token
        return None
    
    def find_subordinate_verbs(self, doc, root_token) -> List[spacy.tokens.Token]:
        """Find verbs that are direct children of the root token."""
        subordinate_verbs = []
        for token in doc:
            if (token.pos_ == "VERB" and 
                token != root_token and 
                len(list(token.ancestors)) >= 1 and 
                root_token in token.ancestors):
                subordinate_verbs.append(token)
        return subordinate_verbs
    
    def get_clause_span(self, verb, doc, all_verbs) -> Tuple[int, int]:
        """Get the token span for a clause centered around a verb."""
        verb_children = list(verb.children)
        if not verb_children:
            return verb.i, verb.i + 1
        
        # Include the verb itself in the span
        token_indices = [verb.i]
        
        # Add all children that aren't verbs themselves
        for child in verb_children:
            if child not in all_verbs:
                token_indices.extend(self._get_subtree_indices(child, all_verbs))
        
        if not token_indices:
            return verb.i, verb.i + 1
            
        return min(token_indices), max(token_indices) + 1
    
    def _get_subtree_indices(self, token, excluded_verbs) -> List[int]:
        """Recursively get all token indices in a subtree, excluding certain verbs."""
        indices = [token.i]
        for child in token.children:
            if child not in excluded_verbs:
                indices.extend(self._get_subtree_indices(child, excluded_verbs))
        return indices
    
    def has_subject(self, clause) -> bool:
        """Check if a clause contains a subject."""
        subject_deps = {"nsubj", "nsubjpass", "csubj", "csubjpass"}
        return any(token.dep_ in subject_deps for token in clause)
    
    def split_into_clauses(self, sentence: str, require_subject: bool = False) -> List[str]:
        """
        Split a sentence into clauses.
        
        Args:
            sentence: The input sentence to split
            require_subject: If True, only return clauses that contain a subject
            
        Returns:
            List of clause texts
        """
        if not sentence.strip():
            return []
            
        doc = self.nlp(sentence)
        
        # Find root and subordinate verbs
        root_token = self.find_root_token(doc)
        if not root_token:
            logger.warning(f"No root token found in sentence: {sentence}")
            return [sentence]
        
        subordinate_verbs = self.find_subordinate_verbs(doc, root_token)
        all_verbs = [root_token] + subordinate_verbs
        
        # Get clause spans
        clause_spans = []
        for verb in all_verbs:
            start, end = self.get_clause_span(verb, doc, all_verbs)
            if start < end:
                clause_spans.append((start, end))
        
        # Extract clauses
        clauses = []
        for start, end in clause_spans:
            clause = doc[start:end]
            if not require_subject or self.has_subject(clause):
                clauses.append(clause.text.strip())
        
        # Remove duplicates while preserving order
        seen = set()
        unique_clauses = []
        for clause in clauses:
            if clause not in seen and clause:
                seen.add(clause)
                unique_clauses.append(clause)
        
        return unique_clauses
    
    def analyze_sentence(self, sentence: str) -> dict:
        """
        Provide detailed analysis of a sentence including clauses and linguistic info.
        
        Returns:
            Dictionary with sentence analysis
        """
        doc = self.nlp(sentence)
        
        analysis = {
            'original_sentence': sentence,
            'clauses': self.split_into_clauses(sentence),
            'tokens': [],
            'dependencies': [],
            'entities': [(ent.text, ent.label_) for ent in doc.ents]
        }
        
        for token in doc:
            token_info = {
                'text': token.text,
                'pos': token.pos_,
                'dep': token.dep_,
                'lemma': token.lemma_,
                'is_verb': token.pos_ == "VERB"
            }
            analysis['tokens'].append(token_info)
            
            if token.dep_ != "ROOT":
                analysis['dependencies'].append(f"{token.text} --{token.dep_}--> {token.head.text}")
        
        return analysis


def main():
    """Main function to demonstrate the enhanced clause splitter."""
    # Test sentences
    test_sentences = [
        "He eats cheese, but he won't eat ice cream.",
        "If it rains later, we won't be able to go to the park.",
        "Although she studied hard, she didn't pass the exam because the questions were too difficult.",
        "The man who lives next door is a doctor.",
        "I think that you should go home."
    ]
    
    splitter = ClauseSplitter()
    
    print("=== Enhanced Clause Splitting Demo ===\n")
    
    for sentence in test_sentences:
        print(f"Original: {sentence}")
        
        # Basic clause splitting
        clauses = splitter.split_into_clauses(sentence)
        print(f"Clauses: {clauses}")
        
        # Detailed analysis
        analysis = splitter.analyze_sentence(sentence)
        print(f"Entities: {analysis['entities']}")
        print("-" * 50)


if __name__ == "__main__":
    main()

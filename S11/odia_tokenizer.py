import os
import glob
import regex as re
from collections import defaultdict, Counter
import numpy as np
from typing import List, Dict, Tuple, Optional

ODIA_PATTERNS = {
    'basic': re.compile(r"""
        \s*[\u0B00-\u0B7F]+  # Basic Odia characters
        |\s*[\p{N}]+         # Numbers
        |\s*[^\s\p{L}\p{N}]+ # Punctuation and symbols
        |\s+                 # Whitespace
        """, re.VERBOSE),
    
    'detailed': re.compile(r"""
        \s*[\u0B00-\u0B7F]+  # Basic Odia characters
        |\s*\u0B3C           # Nukta
        |\s*\u0B3E-\u0B4C    # Dependent vowel signs (matras)
        |\s*\u0B4D           # Virama (halant)
        |\s*[\p{N}]+         # Numbers
        |\s*[।॥]            # Odia punctuation (danda, double danda)
        |\s*[^\s\p{L}\p{N}]+ # Other punctuation
        |\s+                 # Whitespace
        """, re.VERBOSE),
    
    'linguistic': re.compile(r"""
        # Consonant clusters with virama
        \s*[\u0B15-\u0B39]\u0B4D[\u0B15-\u0B39]+  
        # CV combinations (consonant + vowel mark)
        |\s*[\u0B15-\u0B39][\u0B3E-\u0B4C]?      
        # Independent vowels
        |\s*[\u0B05-\u0B14]                       
        # Numbers and punctuation
        |\s*[\p{N}]+                              
        |\s*[^\s\p{L}\p{N}]+
        |\s+
        """, re.VERBOSE)
}

class CompressedOdiaTokenizer:
    def __init__(self, 
                 max_vocab_size: int = 16000,
                 target_compression: float = 4.0,
                 max_token_length: int = 24,
                 pattern_type: str = 'linguistic',
                 max_tokens_per_sequence: Optional[int] = None):
        """
        Initialize the Odia BPE tokenizer
        
        Args:
            max_vocab_size: Maximum vocabulary size (default 16000)
            target_compression: Target compression ratio (default 4.0)
            max_token_length: Maximum token length in characters (default 24)
            pattern_type: Type of tokenization pattern ('basic', 'detailed', or 'linguistic')
            max_tokens_per_sequence: Maximum tokens per sequence (default None)
        """
        self.max_vocab_size = max_vocab_size
        self.target_compression = target_compression
        self.max_token_length = max_token_length
        self.pattern = ODIA_PATTERNS.get(pattern_type, ODIA_PATTERNS['basic'])
        
        # Special tokens
        self.special_tokens = {
            '<UNK>': 0,  # Unknown token
            '<S>': 1,    # Start of text
            '</S>': 2    # End of text
        }
        
        # Vocabulary mappings
        self.stoi: Dict[str, int] = {}  # String to index
        self.itos: Dict[int, str] = {}  # Index to string
        self.merges: Dict[Tuple[int, int], int] = {}  # Merge rules
        
    def _is_odia_char(self, char: str) -> bool:
        """Check if character is in Odia Unicode range"""
        return '\u0B00' <= char <= '\u0B7F'
    
    def _calculate_compression(self, text: str, tokens: List[int]) -> float:
        """
        Calculate compression ratio
        
        Args:
            text: Original text
            tokens: List of token indices
            
        Returns:
            Compression ratio (original size / tokenized size)
        """
        original_size = len(text.encode('utf-8'))
        bits_per_token = np.ceil(np.log2(len(self.stoi)))
        tokenized_size = len(tokens) * np.ceil(bits_per_token / 8)  # Convert bits to bytes
        return original_size / tokenized_size

    def _get_merge_score(self, pair: Tuple[int, int], freq: int, text_len: int) -> float:
        """Calculate merge score for a pair of tokens"""
        print(f"Calculating score for pair: {pair}, freq: {freq}")  # Debug print
        
        def get_str(p):
            print(f"Converting token: {p}, type: {type(p)}")  # Debug print
            if isinstance(p, int):
                token = self.itos[p]
                print(f"Found in itos: {token}, type: {type(token)}")  # Debug print
                if isinstance(token, tuple):
                    result = ''.join(str(t) for t in token)
                    print(f"Converted tuple to string: {result}")  # Debug print
                    return result
                return str(token)
            return str(p)
        
        try:
            # Convert pair to string for length checking
            token_str = ''.join(get_str(p) for p in pair)
            print(f"Combined token string: {token_str}")  # Debug print
            token_len = len(token_str)
            
            # Check length constraint
            if token_len > self.max_token_length:
                print("Token too long")  # Debug print
                return 0.0
            
            # Base score is frequency
            score = freq / text_len
            
            # Bonus for Odia characters
            odia_char_count = sum(1 for c in token_str if '\u0B00' <= c <= '\u0B7F')
            if odia_char_count > 0:
                score *= 1.5
            
            print(f"Final score: {score}")  # Debug print
            return float(score)
        except Exception as e:
            print(f"Error in _get_merge_score: {str(e)}")  # Debug print
            raise

    def _is_common_word_part(self, token_str: str) -> bool:
        """Check if token is likely to be a meaningful word part"""
        # Common Odia word endings
        common_endings = [
            'ର', 'ରେ', 'ଟି', 'ଗୁଡ଼ିକ', 'ମାନେ', 'ଙ୍କୁ', 'ଙ୍କ', 'ଙ୍କର',
            'ଟା', 'ଟାରେ', 'ଗୁଡ଼ାକ', 'ମାନଙ୍କ', 'ମାନଙ୍କୁ'
        ]
        
        # Common Odia word beginnings
        common_beginnings = [
            'ପ୍ର', 'ଅନୁ', 'ଅଧି', 'ପରି', 'ଉପ', 'ସମ', 'ବି', 'ନି', 'ସୁ',
            'ଆ', 'ଇ', 'ଉ', 'ଏ', 'ଓ'
        ]
        
        return (any(token_str.endswith(end) for end in common_endings) or
                any(token_str.startswith(begin) for begin in common_beginnings))

    def _get_stats(self, ids: List[int]) -> Dict[Tuple[int, int], int]:
        """Count token pair frequencies"""
        stats = defaultdict(int)
        for pair in zip(ids, ids[1:]):
            stats[pair] += 1
        return stats

    def _merge(self, ids: List[int], pair: Tuple[int, int], idx: int) -> List[int]:
        """
        Merge all occurrences of pair into a new token
        
        Args:
            ids: List of token indices
            pair: Pair to merge
            idx: New token index
            
        Returns:
            Updated list of token indices
        """
        newids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
                newids.append(idx)
                i += 2
            else:
                newids.append(ids[i])
                i += 1
        return newids

    def pre_tokenize(self, text: str) -> List[str]:
        """
        Pre-tokenize text using regex pattern
        
        Args:
            text: Input text
            
        Returns:
            List of pre-tokenized strings
        """
        tokens = self.pattern.findall(text)
        return [t.strip() for t in tokens if t.strip()]

    def train(self, text: str) -> float:
        """Train the tokenizer on input text"""
        print("Starting training with text length:", len(text))  # Debug print
        
        # Initialize vocabulary with special tokens
        self.stoi = {**self.special_tokens}
        self.itos = {idx: token for token, idx in self.stoi.items()}
        next_idx = len(self.special_tokens)
        print(f"Initialized with {len(self.special_tokens)} special tokens")  # Debug print
        
        # Initial tokenization
        tokens = []
        for match in self.pattern.finditer(text):
            token = match.group(0)
            if token not in self.stoi:
                self.stoi[token] = next_idx
                self.itos[next_idx] = token
                next_idx += 1
            tokens.append(self.stoi[token])
        
        print(f"Initial tokenization complete. Tokens: {len(tokens)}")  # Debug print
        
        # BPE training loop
        while len(self.stoi) < self.max_vocab_size:
            # Count pairs
            pair_freqs = Counter()
            prev_token = None
            for token in tokens:
                if prev_token is not None:
                    pair = (prev_token, token)
                    pair_freqs[pair] += 1
                prev_token = token
            
            print(f"Found {len(pair_freqs)} token pairs")  # Debug print
            
            if not pair_freqs:
                print("No pairs found, breaking")  # Debug print
                break
            
            # Find best pair to merge
            best_score = 0
            best_pair = None
            for pair, freq in pair_freqs.items():
                score = self._get_merge_score(pair, freq, len(text))
                if score > best_score:
                    best_score = score
                    best_pair = pair
            
            if best_pair is None:
                break
            
            # Add merged token to vocabulary
            new_token = best_pair
            self.stoi[new_token] = next_idx
            self.itos[next_idx] = new_token
            self.merges[best_pair] = next_idx
            next_idx += 1
            
            # Update tokens
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i+1]) == best_pair:
                    new_tokens.append(self.stoi[new_token])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
            
            # Check compression ratio
            compression = self._calculate_compression(text, tokens)
            if compression >= self.target_compression:
                break
        
        return self._calculate_compression(text, tokens)

    def encode(self, text: str) -> List[int]:
        """Modified encode method with pre-tokenization"""
        if not self.stoi:
            raise ValueError("Tokenizer needs to be trained first")
            
        # Pre-tokenize the text
        pre_tokens = self.pre_tokenize(text)
        processed_text = ' '.join(pre_tokens)
        
        # Start with characters
        ids = []
        for ch in processed_text:
            if ch in self.stoi:
                ids.append(self.stoi[ch])
            else:
                ids.append(self.stoi['<UNK>'])
                
        # Apply merges in order of creation
        changes_made = True
        while changes_made:
            changes_made = False
            i = 0
            while i < len(ids) - 1:
                current_pair = (ids[i], ids[i+1])
                if current_pair in self.merges:
                    # Replace pair with merged token
                    ids[i:i+2] = [self.merges[current_pair]]
                    changes_made = True
                else:
                    i += 1
                
        return ids

    def decode(self, ids: List[int]) -> str:
        """
        Decode token indices back to text
        
        Args:
            ids: List of token indices
            
        Returns:
            Decoded text
        """
        text = ""
        for idx in ids:
            if idx in self.itos:
                token = self.itos[idx]
                if isinstance(token, tuple):
                    # Recursively decode merged pairs
                    text += self.decode([token[0], token[1]])
                else:
                    text += token
            else:
                text += self.itos[self.special_tokens['<UNK>']]
        return text

    def _is_valid_char(self, char: str) -> bool:
        """Check if character is valid for tokenization"""
        # Valid Odia range
        if '\u0B00' <= char <= '\u0B7F':
            return True
        # Common punctuation and whitespace
        if char in {' ', '.', ',', '।', '?', '!', '\n', '\t'}:
            return True
        # Latin characters and numbers
        if char.isalnum():
            return True
        return False

def load_odia_files(file_pattern: str) -> str:
    """Load all Odia text files matching pattern"""
    text = ""
    for filename in glob.glob(file_pattern):
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
            # Clean the text
            cleaned = ''.join(char for char in content 
                            if '\u0B00' <= char <= '\u0B7F'  # Odia characters
                            or char.isspace()  # Whitespace
                            or char in {'.', ',', '।', '?', '!'})  # Punctuation
            text += cleaned + "\n"
    return text

# Example usage
if __name__ == "__main__":
    # Option 1: Load from files
    input_files_pattern = "odia_texts/*.txt"  # User should put files in this directory
    try:
        text = load_odia_files(input_files_pattern)
        print(f"Loaded text from files matching: {input_files_pattern}")
    except Exception as e:
        print(f"Error loading files: {e}")
        # Fallback to sample text
        text = """
        ଓଡ଼ିଆ ଭାଷା ଏକ ପ୍ରାଚୀନ ଭାରତୀୟ ଭାଷା।
        ଏହା ଭାରତର ଓଡ଼ିଶା ରାଜ୍ୟର ସରକାରୀ ଭାଷା।
        """
        print("Using sample text instead")
    
    # Create and train tokenizer
    tokenizer = CompressedOdiaTokenizer(
        max_vocab_size=16000,
        target_compression=4.0,
        max_token_length=24
    )
    
    # Train
    #print("Training tokenizer...", text)
    compression = tokenizer.train(text)
    print(f"Achieved compression ratio: {compression:.2f}")
    
    # Test encoding/decoding
    tokens = tokenizer.encode(text)
    decoded = tokenizer.decode(tokens)
    print(f"\nVocabulary size: {len(tokenizer.stoi)}")
    print(f"Original text length: {len(text)}")
    print(f"Number of tokens: {len(tokens)}")
    #print("\nOriginal:", text[:200])
    #print("Decoded:", decoded[:200]) 
    
    # Print vocabulary statistics
    print("\nVocabulary Statistics:")
    print(f"Total vocabulary size: {len(tokenizer.stoi)}")
    
    # Print vocabulary with character codes
    print("\nVocabulary Items:")
    for token, idx in sorted(tokenizer.stoi.items(), key=lambda x: x[1]):
        if isinstance(token, tuple):
            # For merged tokens
            token_str = ''.join(str(t) for t in token)
            print(f"ID: {idx}, Token: {token_str} (merged from {token})")
        else:
            # For basic tokens
            if len(token) == 1:
                if token.isprintable():
                    print(f"ID: {idx}, Token: {token} (Unicode: {ord(token):04x})")
                else:
                    print(f"ID: {idx}, Token: <unprintable> (Unicode: {ord(token):04x})")
            else:
                print(f"ID: {idx}, Token: {token}")

    # Print only valid Odia tokens
    print("\nValid Odia Tokens in Vocabulary:")
    for token, idx in sorted(tokenizer.stoi.items(), key=lambda x: x[1]):
        token_str = ''.join(str(t) for t in token) if isinstance(token, tuple) else str(token)
        if any('\u0B00' <= c <= '\u0B7F' for c in token_str) and token_str.isprintable():
            print(f"{idx}: {token_str}")
            
    # Optional: Print some statistics about token lengths
    token_lengths = [len(str(token)) for token in tokenizer.stoi.keys()]
    avg_len = sum(token_lengths) / len(token_lengths)
    print(f"\nAverage token length: {avg_len:.2f} characters")
    print(f"Longest token length: {max(token_lengths)} characters")
    
    # Print only Odia tokens in vocabulary
    print("\nOdia Tokens in Vocabulary:")
    for token, idx in sorted(tokenizer.stoi.items(), key=lambda x: x[1]):
        token_str = ''.join(str(t) for t in token) if isinstance(token, tuple) else str(token)
        if any('\u0B00' <= c <= '\u0B7F' for c in token_str):
            print(f"{idx}: {token_str}") 

    print(f"Achieved compression ratio: {compression:.2f}")
    print(f"Longest token length: {max(token_lengths)} characters")
    print(f"\nVocabulary size: {len(tokenizer.stoi)}")
    print(f"Original text length: {len(text)}")
    print(f"Number of tokens: {len(tokens)}")
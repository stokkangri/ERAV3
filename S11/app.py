import gradio as gr
from odia_tokenizer import CompressedOdiaTokenizer

def analyze_odia_text(text: str) -> str:
    """Process Odia text and return detailed analysis"""
    if not text.strip():
        return "Please enter some Odia text"
    
    try:
        print("Input text:", text[:100], "...")  # Debug print
        
        # Initialize tokenizer
        tokenizer = CompressedOdiaTokenizer(
            max_vocab_size=16000,
            target_compression=4.0,
            max_token_length=24,
            pattern_type='linguistic'
        )
        print("Tokenizer initialized")  # Debug print
        
        # Train on input text
        print("Starting training...")  # Debug print
        compression = tokenizer.train(text)
        print(f"Training complete. Compression: {compression}")  # Debug print
        
        # Get tokens and decode
        print("Encoding text...")  # Debug print
        tokens = tokenizer.encode(text)
        print(f"Text encoded into {len(tokens)} tokens")  # Debug print
        
        print("Decoding tokens...")  # Debug print
        decoded = tokenizer.decode(tokens)
        print("Decoding complete")  # Debug print
        
        # Calculate statistics
        print("Calculating token statistics...")  # Debug print
        token_lengths = [len(str(token)) for token in tokenizer.stoi.keys()]
        print(f"Token types: {[type(token) for token in list(tokenizer.stoi.keys())[:5]]}")  # Debug print
        avg_len = sum(token_lengths) / len(token_lengths)
        
        # Count token types
        odia_tokens = sum(1 for token in tokenizer.stoi.keys() 
                         if any('\u0B00' <= c <= '\u0B7F' 
                         for c in str(token)))
        
        # Generate analysis report
        report = f"""=== Tokenization Summary ===
Input Text Length: {len(text)} characters
Number of Tokens: {len(tokens)}
Vocabulary Size: {len(tokenizer.stoi)}
Compression Ratio: {compression:.2f}

=== Token Statistics ===
Average Token Length: {avg_len:.2f} characters
Maximum Token Length: {max(token_lengths)} characters
Odia Tokens in Vocabulary: {odia_tokens}

=== Sample Analysis ===
First 200 chars of input:
{text[:200]}...

First 200 chars of decoded text:
{decoded[:200]}...

=== First 10 Tokens ==="""
        
        # Modified token display code
        try:
            # Sort by index while handling both string and tuple tokens
            sorted_tokens = sorted(
                tokenizer.stoi.items(), 
                key=lambda x: (x[1] if isinstance(x[1], int) else 0)
            )[:10]
            
            for token, idx in sorted_tokens:
                if isinstance(token, tuple):
                    token_str = ''.join(str(t) for t in token)
                else:
                    token_str = str(token)
                report += f"\n{idx}: {token_str}"
                
        except Exception as e:
            report += f"\nError displaying tokens: {str(e)}"
            
        return report
        
    except Exception as e:
        return f"Error processing text: {str(e)}\nToken types: {[type(token) for token in list(tokenizer.stoi.keys())[:5]]}"

# Create Gradio interface
demo = gr.Interface(
    fn=analyze_odia_text,
    inputs=gr.Textbox(
        lines=5,
        placeholder="Enter Odia text here...",
        label="Input Odia Text"
    ),
    outputs=gr.Textbox(
        lines=15,
        label="Tokenization Analysis"
    ),
    title="Odia Text Tokenizer",
    description="""This tool analyzes Odia text using a custom BPE tokenizer.
    It provides vocabulary statistics, compression ratio, and token examples.""",
    examples=[
        ["ଓଡ଼ିଆ ଭାଷା ଏକ ପ୍ରାଚୀନ ଭାରତୀୟ ଭାଷା।"],
        ["ଏହା ଭାରତର ଓଡ଼ିଶା ରାଜ୍ୟର ସରକାରୀ ଭାଷା।"]
    ]
)

if __name__ == "__main__":
    demo.launch(share=True)

# Text Generation with Transformer using different SpeedUps
- weight sharing (Reduction in parameters)
- torch.set_float32_matmul_precision('high')
- std *= (2 * self.config.n_layer) ** -0.5 ( Solving for residual std scaling issue)
- autocast: with torch.autocast(device_type=device, dtype=torch.bfloat16):
        logits, loss = model(x, y) 
-  torch.compile(model) - only works on Linux
- Flash Attention : y = F.scaled_dot_product_attention(q, k, v, is_causal = True)
- POwer of 2  - Tokens 49152

Use SmolLM2-135 model yaml to find the model parameters.


 
block_size: int = 2048 # max sequence length # max_position_embeddings: 2048
vocab_size: int = 49152 # number of tokens:  #   vocab_size: 49152
n_layer: int = 30 # number of layers #  num_hidden_layers: 30
n_head: int = 9 # number of heads #  num_attention_heads: 9
n_embd: int = 576 # embedding dimension # hidden_size: 576

Tokenizer: tokenizer_name_or_path: HuggingFaceTB/cosmo2-tokenizer

## Usage
- Enter your prompt text in the input box
- Specify how many tokens you want to generate
- Click submit to generate text

## Model Details
- Architecture: Decoder-only Transformer
- Parameters: [147,786,048]
- Training Data: [Tiny Sakesphere - input.txt]
Huggingface app link: //huggingface.co/spaces/stokkangri/tsai_s13_smollm2_135M_param_matched_llm

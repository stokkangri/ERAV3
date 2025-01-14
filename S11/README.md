---
title: Tsai S11 Odia Tokenizer
emoji: 📈
colorFrom: indigo
colorTo: red
sdk: gradio
sdk_version: 5.12.0
app_file: app.py
pinned: false
license: apache-2.0
short_description: Odia language Tokenizer
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

Hugging Face Space: https://huggingface.co/spaces/stokkangri/tsai_s11_odia_tokenizer

# Enhanced Byte Pair Encoding for Odia Text

## Input Data Structure
```
project_root/
├── odia_bpe_tokenizer_enhanced.ipynb
├── data/
│   └── odia/
│       ├── file1.txt
│       └── file2.txt
```
chieved compression ratio: 4.00

Vocabulary size: 8924
Original text length: 11237339
Number of tokens: 579

Average token length: 4.30 characters
Longest token length: 12 characters

Vocabulary Analysis:

Token Type Distribution:
Odia tokens: 7663
Merged tokens: 1257
Special tokens: 3

> Scrape Web to get Odia Text
python3 odia_scraper.py

This will save texts in odia_texts directory.

odia_bpe_tokenizer_enhanced.ipynb is the notebook used to train the tokenizer.

odia_tokenizer.py is the tokenizer implementation.

app.py is the Gradio app for tokenization.
```

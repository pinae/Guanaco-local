# Guanaco-local
Run the Guanaco LLM-Models locally.

## Tests

The file `guanaco7b.py` loads and tests the 
[Guanaco model with 7 billion parameters](https://huggingface.co/timdettmers/guanaco-7b). 
With 4 bit quantization it runs on a RTX2070 Super with only 8GB. 
I tested prompts in english which impressed me. 
Propts in german worked but the model quickly repeated the same sentence.
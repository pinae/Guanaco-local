import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

model_name = "huggyllama/llama-7b"
adapters_name = 'timdettmers/guanaco-7b'

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_4bit=True,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    max_memory={i: '8000MB' for i in range(torch.cuda.device_count())},
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4'
    ),
)
model = PeftModel.from_pretrained(model, adapters_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = "Please summarize the most important aspect in the Works of Immanuel Kant."
formatted_prompt = (
    f"A chat between a curious human and an artificial intelligence assistant."
    f"The assistant gives helpful, detailed, and polite answers to the user's questions.\n"
    f"### Human: {prompt} ### Assistant:"
)
#prompt = "Fasse die wichtigsten Gedanken im Werk von Immanuel Kant zusammen."
#formatted_prompt = (
#    f"Ein Gespräch zwischen einem neugierigen Menschen und einem KI-Assistenten."
#    f"Der KI-Assistant antwortet hilfreich, detailreich und höflich auf die Fragen des Menschen.\n"
#    f"### Mensch: {prompt} ### Assistent:"
#)
inputs = tokenizer(formatted_prompt, return_tensors="pt").to("cuda:0")
outputs = model.generate(inputs=inputs.input_ids, max_new_tokens=1000)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

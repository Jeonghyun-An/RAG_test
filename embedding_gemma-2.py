# pip install accelerate
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_path = "migtissera/Tess-v2.5-Gemma-2-27B-alpha"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path, 
    device_map="auto",
    torch_dtype=torch.bfloat16
)

terminators = [
    tokenizer.convert_tokens_to_ids("<end_of_turn>"),
]


def generate_text(llm_prompt):
    inputs = tokenizer.encode(llm_prompt, add_special_tokens=False, return_tensors="pt")
    input_ids = inputs.to("cuda")
    length = len(input_ids[0])

    instance = {
        "top_p": 1.0,
        "temperature": 0.75,
        "generate_len": 1024,
        "top_k": 50,
    }

    generation = model.generate(
        input_ids, 
        max_length=length + instance["generate_len"],
        use_cache=True,
        do_sample=True,
        top_p=instance["top_p"],
        temperature=instance["temperature"],
        top_k=instance["top_k"],
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=terminators,
    )
    # rest= tokenizer.decode(generation[0])
    output = generation[0][length:]
    string = tokenizer.decode(output, skip_special_tokens=True)
    return f"{string}"

conversation = f"""<bos><start_of_turn>user\n"""

while True:
    user_input = input("You: ")
    llm_prompt = f"{conversation}{user_input}<end_of_turn>\n<start_of_turn>model\n"
    answer = generate_text(llm_prompt)
    print(answer)
    conversation = f"{llm_prompt}{answer}\n<start_of_turn>user\n"

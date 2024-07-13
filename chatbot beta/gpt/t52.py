# First, install SentencePiece if not already installed
# !pip install sentencepiece

from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load the tokenizer and model
model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

def generate_response(input_text):
    # Tokenize the input text
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    
    # Generate the response
    outputs = model.generate(input_ids)
    
    # Decode the output to text
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Example conversation
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break
    
    bot_response = generate_response(user_input)
    print(f"Bot: {bot_response}")

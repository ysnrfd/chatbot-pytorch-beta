import json
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load intents and preprocess data as before
with open('intents.json', 'r') as f:
    intents = json.load(f)

# Initialize GPT-2 tokenizer and model
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Save the model and tokenizer to a directory
model.save_pretrained('gpt2-small')
tokenizer.save_pretrained('gpt2-small')

# Load the model and tokenizer from the saved directory
tokenizer = GPT2Tokenizer.from_pretrained('gpt2-small')
model = GPT2LMHeadModel.from_pretrained('gpt2-small')

# Function to generate response using GPT-2 Small
def generate_response(input_text):
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    output = model.generate(input_ids, max_length=100, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Interactive chat session
print("Start chatting with the chatbot (type 'quit' to exit)")

while True:
    try:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            break

        # Generate response using GPT-2 Small
        response = generate_response(user_input)
        print("Chatbot:", response)

    except KeyboardInterrupt:
        print("\nInterrupted by user. Exiting...")
        break
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        continue

from transformers import DistilBertTokenizer, DistilBertModel
import torch

# Load tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained('distilbert-base-uncased')

# Function to process input and generate response
def generate_response(input_text):
    # Tokenize input
    encoded_input = tokenizer(input_text, return_tensors='pt')

    # Model inference
    with torch.no_grad():
        outputs = model(**encoded_input)
        last_hidden_state = outputs.last_hidden_state  # Contextual embeddings
        pooled_output = outputs.pooler_output  # Pooled representation of the entire input sequence

    # Dummy response generation (replace with actual logic)
    response = "Sure, I can help with that."

    return response

# Interaction loop
print("Chatbot: Hi there! How can I assist you today?")
while True:
    user_input = input("You: ")

    # Exit condition
    if user_input.lower() == 'exit':
        print("Chatbot: Goodbye!")
        break

    # Generate response
    response = generate_response(user_input)
    print("Chatbot:", response)

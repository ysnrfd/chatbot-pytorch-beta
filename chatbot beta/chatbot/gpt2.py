import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Load pretrained GPT-2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Move model to appropriate device (CPU or GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Chatbot function
def chatbot(input_text):
    try:
        # Tokenize input text
        inputs = tokenizer(input_text, return_tensors='pt')
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        
        # Generate response
        with torch.no_grad():
            output = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=100, num_return_sequences=1)
        
        # Decode and return response
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        return response
    
    except Exception as e:
        print(f"Error: {e}")
        return "I'm sorry, I couldn't understand that."

# Interaction loop
print("Welcome to the GPT-2 Chatbot. You can start chatting now!")
while True:
    # Take user input
    user_input = input("You: ")
    
    # Exit condition
    if user_input.lower() == 'exit':
        print("Chatbot: Goodbye!")
        break
    
    # Get chatbot response
    chatbot_response = chatbot(user_input)
    
    # Print chatbot response
    print("Chatbot:", chatbot_response)

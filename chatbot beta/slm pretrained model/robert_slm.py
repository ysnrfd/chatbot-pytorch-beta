from transformers import RobertaTokenizer, RobertaForMaskedLM
import torch

# Load the tokenizer and model
model_name = "roberta-base"
tokenizer = RobertaTokenizer.from_pretrained(model_name)
model = RobertaForMaskedLM.from_pretrained(model_name)

def generate_response(input_text):
    # Tokenize the input text and add the mask token at the end
    input_text_with_mask = input_text + " <mask>."
    input_ids = tokenizer.encode(input_text_with_mask, return_tensors="pt")

    # Generate prediction for the masked token
    with torch.no_grad():
        outputs = model(input_ids)
        predictions = outputs.logits

    # Get the predicted token
    masked_index = (input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
    predicted_token_id = predictions[0, masked_index].argmax(axis=-1)
    predicted_token = tokenizer.decode(predicted_token_id)

    # Replace the mask token with the predicted token
    response = input_text_with_mask.replace(tokenizer.mask_token, predicted_token)
    return response

# Example conversation
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break
    
    bot_response = generate_response(user_input)
    print(f"Bot: {bot_response}")

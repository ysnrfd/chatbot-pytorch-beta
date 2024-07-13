import json
import numpy as np
import torch
import spacy
import random
from nltk.corpus import stopwords
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim

# Load intents and preprocess data as before
with open('intents_bio.json', 'r') as f:
    intents = json.load(f)

nlp = spacy.load('en_core_web_md')
stop_words = set(stopwords.words('english'))

all_words = []
tags = []
xy = []

for intent in intents['intents']:
    for pattern in intent['patterns']:
        doc = nlp(pattern.lower())
        words = [token.lemma_ for token in doc if token.text.lower() not in stop_words]
        all_words.extend(words)
        xy.append((words, intent['tag']))
        if intent['tag'] not in tags:
            tags.append(intent['tag'])

all_words = sorted(set(all_words))
tags = sorted(set(tags))

# Define or Load the necessary data (X, y) for chatbot interaction
# Example of loading the saved model and necessary data
class ChatDataset(Dataset):
    def __init__(self, X, y):
        self.n_samples = len(X)
        self.X_data = torch.tensor(X, dtype=torch.float32)
        self.y_data = torch.tensor(y, dtype=torch.long)

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples

# Load trained model
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc3(out)
        return out

# Load the model
model = NeuralNet(len(all_words), 32, len(tags))
model.load_state_dict(torch.load('chatbot_model.pth'))
model.eval()

# Define function to process user input and generate response
def process_input(user_input, model, all_words, tags, intents):
    model.eval()
    with torch.no_grad():
        doc = nlp(user_input.lower())
        words = [token.lemma_ for token in doc if token.text.lower() not in stop_words]
        bag = [1 if w in words else 0 for w in all_words]
        input_data = torch.tensor(bag, dtype=torch.float32).unsqueeze(0)
        output = model(input_data)
        _, predicted = torch.max(output, dim=1)
        tag = tags[predicted.item()]

        # Softmax to calculate probability
        probs = torch.softmax(output, dim=1)
        confidence = probs[0][predicted.item()].item()

        # Threshold to define confidence
        if confidence > 0.75:  # Adjust confidence threshold as needed
            for intent in intents['intents']:
                if tag == intent['tag']:
                    response = random.choice(intent['responses'])
                    break
        else:
            response = "I'm sorry, I don't understand. Could you please rephrase?"

        return response

# Interactive chat session
print("Start chatting with the chatbot (type 'quit' to exit)")
context = {}

while True:
    try:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            break

        # Process user input
        response = process_input(user_input, model, all_words, tags, intents)
        print("Chatbot:", response)

        # Update context or session state if needed
        # Example: context['last_intent'] = predicted_intent

    except KeyboardInterrupt:
        print("\nInterrupted by user. Exiting...")
        break
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        continue

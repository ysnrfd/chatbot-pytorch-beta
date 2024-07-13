import json
import torch
import spacy
import random
from nltk.corpus import stopwords
from torch import nn, optim

# Load intents and preprocess data
with open('intents.json', 'r') as f:
    intents = json.load(f)

nlp = spacy.load('en_core_web_md')
stop_words = set(stopwords.words('english'))

all_words = []
tags = []
xy = []

# Tokenize patterns and collect tags
for intent in intents['intents']:
    for pattern in intent['patterns']:
        doc = nlp(pattern.lower())
        words = [token.lemma_ for token in doc if token.text.lower() not in stop_words]
        all_words.extend(words)
        xy.append((words, intent['tag']))
        if intent['tag'] not in tags:
            tags.append(intent['tag'])

# Prepare unique word and tag lists
all_words = sorted(set(all_words))
tags = sorted(set(tags))

# Define a mapping from words to indices
word_to_index = {word: idx for idx, word in enumerate(all_words)}
tag_to_index = {tag: idx for idx, tag in enumerate(tags)}

# Prepare training data
X = []
y = []
for (words, intent_tag) in xy:
    bag = [1 if word in words else 0 for word in all_words]
    X.append(bag)
    y.append(tag_to_index[intent_tag])

# Convert to PyTorch tensors
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

# Define Neural Network model
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

# Define hyperparameters
input_size = len(all_words)
hidden_size = 32
output_size = len(tags)
learning_rate = 0.001
num_epochs = 1000

# Initialize model, loss function, and optimizer
model = NeuralNet(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X)
    loss = criterion(outputs, y)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Save trained model
torch.save(model.state_dict(), 'chatbot_model.pth')

# Define function to process user input and generate response
def process_input(user_input, model, all_words, tags, intents):
    model.eval()
    with torch.no_grad():
        # Tokenize user input
        doc = nlp(user_input.lower())
        words = [token.lemma_ for token in doc if token.text.lower() not in stop_words]
        bag = [1 if word in words else 0 for word in all_words]
        input_data = torch.tensor(bag, dtype=torch.float32).unsqueeze(0)

        # Predict intent
        output = model(input_data)
        _, predicted = torch.max(output, dim=1)
        tag = tags[predicted.item()]

        # Generate response based on confidence
        probs = torch.softmax(output, dim=1)
        confidence = probs[0][predicted.item()].item()

        if confidence > 0.75:
            for intent in intents['intents']:
                if tag == intent['tag']:
                    response = random.choice(intent['responses'])
                    break
        else:
            response = "I'm sorry, I don't understand. Could you please rephrase?"

        return response

# Interactive chat session
print("Start chatting with the chatbot (type 'quit' to exit)")

while True:
    try:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            break

        # Process user input and generate response
        response = process_input(user_input, model, all_words, tags, intents)
        print("Chatbot:", response)

    except KeyboardInterrupt:
        print("\nInterrupted by user. Exiting...")
        break
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        continue

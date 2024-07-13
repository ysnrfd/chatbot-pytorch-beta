import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import spacy
from nltk.corpus import stopwords
import random
from sklearn.model_selection import train_test_split

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

# Split data into training and validation sets
X = []
y = []
for (pattern_words, tag) in xy:
    bag = [1 if w in pattern_words else 0 for w in all_words]
    X.append(bag)
    y.append(tags.index(tag))

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Define PyTorch dataset and dataloaders
class ChatDataset(Dataset):
    def __init__(self, X, y):
        self.n_samples = len(X)
        self.X_data = torch.tensor(X, dtype=torch.float32)
        self.y_data = torch.tensor(y, dtype=torch.long)

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples

train_dataset = ChatDataset(X_train, y_train)
val_dataset = ChatDataset(X_val, y_val)

train_loader = DataLoader(dataset=train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=8, shuffle=False)

# Define neural network architecture
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

# Hyperparameters
input_size = len(all_words)
hidden_size = 32
output_size = len(tags)
learning_rate = 0.001
num_epochs = 2000

# Initialize model, loss function, and optimizer
model = NeuralNet(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop with validation
for epoch in range(num_epochs):
    model.train()
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

    if (epoch+1) % 50 == 0:
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
                val_loss += criterion(outputs, targets).item()

        val_loss /= len(val_loader)
        accuracy = correct / total * 100.0

        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Train Loss: {loss.item():.4f}, '
              f'Val Loss: {val_loss:.4f}, '
              f'Val Accuracy: {accuracy:.2f}%')

print('Training complete.')

# Save the model
torch.save(model.state_dict(), 'chatbot_model.pth')
print('Model saved.')

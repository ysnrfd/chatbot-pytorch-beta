import json
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Step 1: Load intents file
with open('intents.json', 'r') as file:
    intents = json.load(file)

# Step 2: Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Step 3: Initialize lists for training data
words = []
classes = []
documents = []
ignore_words = list(string.punctuation)

# Step 4: Loop through intents and preprocess data
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Tokenize and lemmatize words
        w = nltk.word_tokenize(pattern)
        w = [lemmatizer.lemmatize(word.lower()) for word in w if word not in stop_words and word not in ignore_words]
        words.extend(w)
        documents.append((w, intent['tag']))
        # Add intent tag to classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Step 5: Sort and deduplicate words and classes
words = sorted(set(words))
classes = sorted(set(classes))

# Step 6: Print summary of loaded intents
print(len(documents), "documents")
print(len(classes), "classes", classes)
print(len(words), "unique lemmatized words", words)

# Step 7: Create training data using Bag of Words approach
training = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = []
    pattern_words = doc[0]
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])

# Step 8: Shuffle training data and convert to numpy array
random.shuffle(training)
training = np.array(training)

# Step 9: Separate features and labels
train_x = list(training[:,0])
train_y = list(training[:,1])

# Step 10: Define PyTorch dataset and dataloader
class ChatDataset(Dataset):
    def __init__(self, X, y):
        self.n_samples = len(X)
        self.X_data = torch.tensor(X, dtype=torch.float32)
        self.y_data = torch.tensor(y, dtype=torch.float32)

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples

dataset = ChatDataset(train_x, train_y)
train_loader = DataLoader(dataset=dataset, batch_size=8, shuffle=True)

# Step 11: Define neural network architecture
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

# Step 12: Hyperparameters and training setup
input_size = len(train_x[0])
hidden_size = 8
output_size = len(train_y[0])
learning_rate = 0.001
num_epochs = 1000

model = NeuralNet(input_size, hidden_size, output_size)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Step 13: Training the model
for epoch in range(num_epochs):
    for (inputs, targets) in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

print('Training complete.')

# Step 14: Save the model
torch.save(model.state_dict(), 'chatbot_model.pth')
print('Model saved.')

# Step 15: Function to predict intent from user input
def predict_intent(sentence, model, words, classes):
    model.eval()
    with torch.no_grad():
        # Tokenize and lemmatize user input
        sentence_words = nltk.word_tokenize(sentence)
        sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
        
        # Create bag of words
        bow = [0]*len(words)
        for s in sentence_words:
            for i, w in enumerate(words):
                if w == s:
                    bow[i] = 1
        
        # Convert to tensor
        bow_tensor = torch.tensor(bow, dtype=torch.float32).unsqueeze(0)
        
        # Predict using model
        output = model(bow_tensor)
        _, predicted = torch.max(output, 1)
        
        # Retrieve intent tag
        tag = classes[predicted.item()]
        
        # Select random response from intents file
        for intent in intents['intents']:
            if intent['tag'] == tag:
                return random.choice(intent['responses'])

# Step 16: Interactive chat session
print("Start chatting with the chatbot (type 'quit' to exit)")

while True:
    user_input = input("You: ")
    if user_input.lower() == 'quit':
        break

    response = predict_intent(user_input, model, words, classes)
    print("Chatbot:", response)

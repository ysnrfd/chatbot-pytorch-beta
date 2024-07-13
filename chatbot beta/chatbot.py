import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import random

# Ensure necessary NLTK data is available
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Step 1: Load and preprocess data
with open('intents.json', 'r') as f:
    intents = json.load(f)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

corpus = []
tags = []
xy = []

for intent in intents['intents']:
    for pattern in intent['patterns']:
        words = word_tokenize(pattern.lower())
        words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
        sentence = " ".join(words)
        corpus.append(sentence)
        xy.append((sentence, intent['tag']))
        if intent['tag'] not in tags:
            tags.append(intent['tag'])

# Step 2: TF-IDF Vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus).toarray()

# Encode tags
tags = sorted(set(tags))
tag_to_idx = {tag: idx for idx, tag in enumerate(tags)}
y = np.array([tag_to_idx[tag] for _, tag in xy])

# Step 3: Define PyTorch dataset and dataloader
class ChatDataset(Dataset):
    def __init__(self, X, y):
        self.n_samples = len(X)
        self.X_data = torch.tensor(X, dtype=torch.float32)
        self.y_data = torch.tensor(y, dtype=torch.long)

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples

dataset = ChatDataset(X, y)
train_loader = DataLoader(dataset=dataset, batch_size=8, shuffle=True)

# Step 4: Define neural network architecture
class LSTMNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMNet, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        out, _ = self.lstm(x.unsqueeze(1))
        out = self.fc(self.dropout(self.relu(out[:, -1, :])))
        return out

# Step 5: Hyperparameters and training setup
input_size = X.shape[1]
hidden_size = 64
output_size = len(tags)
learning_rate = 0.001
num_epochs = 2000

model = LSTMNet(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.1)  # Learning rate scheduler

# Step 6: Training the model
for epoch in range(num_epochs):
    for (inputs, targets) in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
    
    scheduler.step()  # Adjust learning rate

    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

print('Training complete.')

# Step 7: Save the model
torch.save(model.state_dict(), 'chatbot_model.pth')
print('Model saved.')

# Step 8: Function to predict intent from user input
def predict_intent(sentence, model, vectorizer, tags):
    model.eval()
    with torch.no_grad():
        words = word_tokenize(sentence.lower())
        words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
        sentence = " ".join(words)
        bag = vectorizer.transform([sentence]).toarray()
        input_data = torch.tensor(bag, dtype=torch.float32)
        output = model(input_data)
        _, predicted = torch.max(output, dim=1)
        tag = tags[predicted.item()]

        # Softmax to calculate probability
        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]

        # Threshold to define confidence
        if prob.item() > 0.75:
            for intent in intents['intents']:
                if tag == intent['tag']:
                    return random.choice(intent['responses'])
        else:
            return "I'm sorry, I don't understand."

# Step 9: Interactive chat session
print("Start chatting with the chatbot (type 'quit' to exit)")
while True:
    user_input = input("You: ")
    if user_input.lower() == 'quit':
        break

    response = predict_intent(user_input, model, vectorizer, tags)
    print("Chatbot:", response)

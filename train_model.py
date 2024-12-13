import pandas
import string
import numpy as np
from torch.utils.data import Dataset
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import time

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# turns the text chunks into an object. Not fully sure how this part works or why it's its own class, but I assume there must be some funtions from its parent class Dataset that are needed
class TextDataset(Dataset):
    def __init__(self, text_chunks):
        self.text_chunks = text_chunks

    def __len__(self):
        return len(self.text_chunks)

    def __getitem__(self, idx):
        text_chunk = self.text_chunks[idx]
        return text_chunk[:-1].long(), text_chunk[1:].long()


# creates a model with no hidden layer that passes attribute values in after embedding but before the LSTM layer
class RNNone(nn.Module):
    def __init__(self, vocab_size, embed_dim, rnn_hidden_size, attributes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim) # embedding layer
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn = nn.LSTM(embed_dim + len(attributes), rnn_hidden_size, batch_first=True) # rnn LSTM layer
        self.fc = nn.Linear(rnn_hidden_size, vocab_size) # fully connected (output) layer
        self.attributes = attributes

    def forward(self, x, hidden, cell): # passes the outputs forward to the next layer
        out = self.embedding(x).unsqueeze(1)
        out = torch.cat(out, self.attributes)
        out, (hidden, cell) = self.rnn(out, (hidden, cell))
        out = self.fc(out).reshape(out.size(0), -1)
        return out, hidden, cell

    def init_hidden(self, batch_size): # initializes the hidden state with zeros
        hidden = torch.zeros(1, batch_size, self.rnn_hidden_size)
        cell = torch.zeros(1, batch_size, self.rnn_hidden_size)
        return hidden.to(DEVICE), cell.to(DEVICE)


# creates a model with no hidden layer that passes attribute values in after the LSTM layer and before the output layer
class RNNtwo(nn.Module):
    def __init__(self, vocab_size, embed_dim, rnn_hidden_size, attributes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim) # embedding layer
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn = nn.LSTM(embed_dim, rnn_hidden_size, batch_first=True) # rnn LSTM layer
        self.fc = nn.Linear(rnn_hidden_size + len(attributes), vocab_size) # fully connected (output) layer
        self.attributes = attributes

    def forward(self, x, hidden, cell): # passes the outputs forward to the next layer
        out = self.embedding(x).unsqueeze(1)
        out, (hidden, cell) = self.rnn(out, (hidden, cell))
        out = torch.cat(out, self.attributes)
        out = self.fc(out).reshape(out.size(0), -1)
        return out, hidden, cell

    def init_hidden(self, batch_size): # initializes the hidden state with zeros
        hidden = torch.zeros(1, batch_size, self.rnn_hidden_size)
        cell = torch.zeros(1, batch_size, self.rnn_hidden_size)
        return hidden.to(DEVICE), cell.to(DEVICE)


# creates a model WITH a hidden layer that passes attribute values in after the LSTM layer and before the hidden layer
# I'VE NOT FINISHED IMPLEMENTING THIS --> someone will have to finish the self.hidden = nn. line that I started writing
# Documentation for torch.nn should give you the function for a hidden layer 
class RNNthree(nn.Module):
    def __init__(self, vocab_size, embed_dim, rnn_hidden_size, attributes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim) # embedding layer
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn = nn.LSTM(embed_dim, rnn_hidden_size, batch_first=True) # rnn LSTM layer
        self.hidden = nn.Linear(rnn_hidden_size + len(attributes), rnn_hidden_size)
        self.fc = nn.Linear(rnn_hidden_size, vocab_size) # fully connected (output) layer
        self.attributes = attributes

    def forward(self, x, hidden, cell): # passes the outputs forward to the next layer
        out = self.embedding(x).unsqueeze(1)
        out, (hidden, cell) = self.rnn(out, (hidden, cell))
        out = torch.cat(out, self.attributes)
        out = self.hidden(out) # THIS LINE MIGHT NEED TO CHANGE, DEPENDING ON HOW TO IMPLEMENT A HIDDEN LAYER
        out = self.fc(out).reshape(out.size(0), -1)
        return out, hidden, cell

    def init_hidden(self, batch_size): # initializes the hidden state with zeros
        hidden = torch.zeros(1, batch_size, self.rnn_hidden_size)
        cell = torch.zeros(1, batch_size, self.rnn_hidden_size)
        return hidden.to(DEVICE), cell.to(DEVICE)



# creates a model WITH a hidden layer that passes attribute values in after the hidden layer and before the ouput layer
# SAME AS ABOVE --> IVE NOT FINISHED IMPLEMENTING THIS
class RNNfour(nn.Module):
    def __init__(self, vocab_size, embed_dim, rnn_hidden_size, attributes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim) # embedding layer
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn = nn.LSTM(embed_dim, rnn_hidden_size, batch_first=True) # rnn LSTM layer
        self.hidden = nn.Linear(rnn_hidden_size, rnn_hidden_size)
        self.fc = nn.Linear(rnn_hidden_size + len(attributes), vocab_size) # fully connected (output) layer
        self.attributes = attributes

    def forward(self, x, hidden, cell): # passes the outputs forward to the next layer
        out = self.embedding(x).unsqueeze(1)
        out, (hidden, cell) = self.rnn(out, (hidden, cell))
        # out, (hidden, cell) = self.hidden(out, (hidden, cell)) # THIS LINE MIGHT NEED TO CHANGE, DEPENDING ON HOW TO IMPLEMENT A HIDDEN LAYER
        out = self.hidden(out)
        print('Out is:')
        print(out)
        print('Attributes are:')
        print(self.attributes)

        out = torch.cat(out, self.attributes)
        out = self.fc(out).reshape(out.size(0), -1)
        return out, hidden, cell

    def init_hidden(self, batch_size): # initializes the hidden state with zeros
        hidden = torch.zeros(1, batch_size, self.rnn_hidden_size)
        cell = torch.zeros(1, batch_size, self.rnn_hidden_size)
        return hidden.to(DEVICE), cell.to(DEVICE)



# scales predicted probabilities from our model to make the predictions more random, depending on the parameters
def top_p_sampling(logits, temperature=.8, top_p=0.8):
    # Ensure logits are a PyTorch tensor and move to DEVICE

    # Apply temperature scaling
    scaled_logits = logits / temperature

    # Convert logits to probabilities using softmax
    probabilities = torch.softmax(scaled_logits, dim=-1)

    # Sort probabilities and compute cumulative sum
    sorted_indices = torch.argsort(probabilities, descending=True)
    sorted_probabilities = probabilities[sorted_indices]
    cumulative_probabilities = torch.cumsum(sorted_probabilities, dim=-1)

    # Apply top-p filtering
    indices_to_keep = cumulative_probabilities <= top_p
    truncated_probabilities = sorted_probabilities[indices_to_keep]

    # Rescale the probabilities
    truncated_probabilities /= torch.sum(truncated_probabilities)

    # Convert to numpy arrays for random choice
    truncated_probabilities = truncated_probabilities.cpu().numpy()
    sorted_indices = sorted_indices.cpu().numpy()
    indices_to_keep = indices_to_keep.cpu().numpy()

    # Sample from the truncated distribution
    if not indices_to_keep.any():
        # Handle the empty case - for example, using regular sampling without top-p
        probabilities = torch.softmax(logits / temperature, dim=-1)
        next_word_index = torch.multinomial(probabilities, 1).item()
    else:
        # Existing sampling process
        next_word_index = np.random.choice(sorted_indices[indices_to_keep], p=truncated_probabilities)

    return torch.tensor(next_word_index).to(DEVICE)


# turns a string (I think) into a list of each unique word with periods included
# NEED TO CHANGE THIS FUNCTION SO THAT RATHER THAN INCLUDING WORDS AND PERIODS, IT INCLUDES WORDS AND THE NEWLINE CHARACTER
def tokenize(doc):
    # Exclude period from the punctuation list
    punctuation_to_remove = string.punctuation.replace('.', '')

    # Create translation table that removes specified punctuation except period
    table = str.maketrans('', '', punctuation_to_remove)

    tokens = doc.split()
    # Further split tokens by period and keep periods as separate tokens
    split_tokens = []
    for token in tokens:
        split_tokens.extend(token.replace('.', ' .').split())

    tokens = [w.translate(table) for w in split_tokens]
    tokens = [word for word in tokens if word.isalpha() or word == '.']
    tokens = [word.lower() for word in tokens]

    return tokens


# uses the model and a seed string to generate lyrics
# AS OF RIGHT NOW IS NOT CONSIDERING ANY OF THE ATTRIBUTE VALUES
def generate(model, seed_str, len_generated_text=50, temperature=.8, top_p=0.8):

    seed_tokens = tokenize(seed_str)

    encoded_input = torch.tensor([word2int[t] for t in seed_tokens])
    encoded_input = torch.reshape(encoded_input, (1, -1)).to(DEVICE)

    generated_str = seed_str

    model.eval()
    with torch.inference_mode():
      hidden, cell = model.init_hidden(1)
      hidden = hidden.to(DEVICE)
      cell = cell.to(DEVICE)
      for w in range(len(seed_tokens)-1):
          _, hidden, cell = model(encoded_input[:, w].view(1), hidden, cell)

      last_word = encoded_input[:, -1]
      for i in range(len_generated_text):
          logits, hidden, cell = model(last_word.view(1), hidden, cell)
          logits = torch.squeeze(logits, 0)
          last_word = top_p_sampling(logits, temperature, top_p)
          generated_str += " " + str(word_array[last_word])

    return generated_str.replace(" . ", ". ")



def main():

    start_time = time.time()

    print('-------------------------------------------')
    print('Opening and parsing dataset')
    print('Time: ' + str(time.time() - start_time))
    print('-------------------------------------------')


    ### loads the dataset
    dataset = pandas.read_csv("songs_cleaned.csv")
    dataset['lyrics_y']

    ### removes the attributes we don't want to use, turns the ones we want into a tensor
    weird_attributes = ['world/life', 'shake the audience', 'family/gospel', 'communication', 'music', 'like/girls', 'feelings', 'loudness', 'instrumentalness', 'valence']
    dataset.drop(weird_attributes, axis=1)

    # turn the attributes that we want into a tensor
    good_attributes = ['dating', 'violence', 'romantic', 'obscene', 'sadness', 'danceability', 'energy', 'acousticness', 'night/time', 'movement/places', 'light/visual perceptions', 'family/spiritual']
    attribute_data = dataset[good_attributes]
    attribute_data = torch.from_numpy(attribute_data.values).float()

    ### removes all punctuation from the lyrics (need to change this so that \n newline characters aren't removed too tho
    # set so that only words that aren't already in the set can get added
    unique_words = set()
    tokens = []

    i=0
    for lyrics in dataset['lyrics_y']:
        if isinstance(lyrics, str):

            # split the lyric string into individual word strings
            words = lyrics.split() 
            for word in words:
                # good word is the word after punctuation has been removed
                goodword = ''.join(char for char in word if char not in string.punctuation)

                unique_words.add(goodword.lower())
                tokens.append(goodword.lower())
            i += 1
    vocab = sorted(unique_words) # vocab is the set of every unique word in the dataset


    ### gives every unique word a unique number that represents that word
    word2int = {word:i for i, word in enumerate(vocab)} # makes a dictionary with a corresponding number for each unique word
    word_array = np.array(vocab) # numpy array of all possible words

    # encodes the text from the lyrics as numbers
    text_encoded = np.array(
        [word2int[word] for word in tokens],
        dtype=np.int32
    )


    ### a sequence/chunk is the a list of so many words, which is used to make a list of previously seen words/the next word to come
    seq_length = 8
    chunk_size = 9
    batch_size = 64 # I think this means that 64 text chunks (where each text chunk is chunk_size words) are used, not that 64 different songs are used

    # makes chunks for all of the lyrics of however many words in a row
    text_chunks = [text_encoded[i:i+chunk_size] for i in range(len(text_encoded)-chunk_size+1)]

    # # for every sequence, break the chunk up into the previously seen words and the target word
    # for seq in text_chunks[:1]:
    #     input_seq = seq[:seq_length]
    #     target = seq[seq_length]

    # turning the text chunks into a dataset we can use
    seq_dataset = TextDataset(torch.tensor(text_chunks))

    # i don't know what this is doing
    seq_dl = DataLoader(seq_dataset, batch_size=batch_size, shuffle=True, drop_last=True) # don't know what's happening with DataLoader


    ### creating the model

    print('-------------------------------------------')
    print('Creating model')
    print('Time: ' + str(time.time() - start_time))
    print('-------------------------------------------')

    # vocab size is used to create the size of the embedding layer
    vocab_size = len(word_array)
    # i THINK this is how many outputs the embedding layer has. So, the embedding layer maps however many unique words we have into 256 values
    embed_dim = 256
    rnn_hidden_size = 512 # THIS IS ACTUALLY THE LSTM LAYER SIZE
    # hidden_layer_size = 

    # initialize our model
    model = RNNfour(vocab_size, embed_dim, rnn_hidden_size, attribute_data)

    # create GPU device and move the model to it if the GPU is available
    model = model.to(DEVICE)


    ### create our loss function/optimizer, then train the model
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    # WANT TO LOOK AT HOW MANY BATCHES WE HAVE TOTAL, AND AT A MINIMUM WE NEED AT LEAST THAT MANY EPOCHS
    num_epochs = 1000 # 1 epoch is just through one batch, not every single instance

    print('-------------------------------------------')
    print('Training the model')
    print('Time: ' + str(time.time() - start_time))
    print('-------------------------------------------')

    # training the model
    model.train()
    for epoch in range(num_epochs):
        hidden, cell = model.init_hidden(batch_size)
        seq_batch, target_batch = next(iter(seq_dl))
        seq_batch = seq_batch.to(DEVICE)
        target_batch = target_batch.to(DEVICE)
        optimizer.zero_grad()
        loss = 0 
        for w in range(seq_length):
            pred, hidden, cell = model(seq_batch[:, w], hidden, cell) # passing whichever batch in to the model with the hidden and cell state
            loss += loss_fn(pred, target_batch[:, w])
        loss.backward()
        optimizer.step()
        loss = loss.item()/seq_length
        if epoch % 10 == 0:
            print(f'Epoch {epoch} loss: {loss:.4f}')


    ### generate a new string
    generate(model, seed_str="I was")

if __name__ == "__main__":
    main()
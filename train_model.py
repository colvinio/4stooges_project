import pandas
import string
import numpy as np
from torch.utils.data import Dataset
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import time
import random

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
        self.rnn = nn.LSTM(embed_dim + attributes, rnn_hidden_size, batch_first=True) # rnn LSTM layer
        self.fc = nn.Linear(rnn_hidden_size, vocab_size) # fully connected (output) layer

    def forward(self, x, hidden, cell, attributes): # passes the outputs forward to the next layer
        out = self.embedding(x).unsqueeze(1)
        out = torch.cat(out, attributes)
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
        self.fc = nn.Linear(rnn_hidden_size + attributes, vocab_size) # fully connected (output) layer

    def forward(self, x, hidden, cell, attributes): # passes the outputs forward to the next layer
        out = self.embedding(x).unsqueeze(1)
        out, (hidden, cell) = self.rnn(out, (hidden, cell))
        out = torch.cat(out, attributes)
        out = self.fc(out).reshape(out.size(0), -1)
        return out, hidden, cell

    def init_hidden(self, batch_size): # initializes the hidden state with zeros
        # BATCH SIZE IS HOW MANY SEQUENCE --> TARGET PAIRS THE MODEL WILL BE RECEIVING FOR THAT SONG
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
        self.hidden = nn.Linear(rnn_hidden_size + attributes, rnn_hidden_size)
        self.fc = nn.Linear(rnn_hidden_size, vocab_size) # fully connected (output) layer

    def forward(self, x, hidden, cell, attributes): # passes the outputs forward to the next layer
        out = self.embedding(x).unsqueeze(1)
        out, (hidden, cell) = self.rnn(out, (hidden, cell))
        out = torch.cat(out, attributes)
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
        self.fc = nn.Linear(rnn_hidden_size + attributes, vocab_size) # fully connected (output) layer

    def forward(self, x, hidden, cell, attributes): # passes the outputs forward to the next layer
        out = self.embedding(x).unsqueeze(1)
        out, (hidden, cell) = self.rnn(out, (hidden, cell))
        # out, (hidden, cell) = self.hidden(out, (hidden, cell)) # THIS LINE MIGHT NEED TO CHANGE, DEPENDING ON HOW TO IMPLEMENT A HIDDEN LAYER
        out = self.hidden(out)
        # print("----------------------------")
        # print("----------------------------")
        # print('Out is:')
        # print(out)
        # print(out.shape)
        # print("----------------------------")
        expanded_attributes = attributes.view(1, 1, -1).expand(out.shape[0], 1, -1)
        # print('Attributes are:')
        # print(expanded_attributes)
        # print(expanded_attributes.shape)
        # print("----------------------------")
        # print("----------------------------")

        out = torch.cat((out, expanded_attributes), dim=2)
        out = self.fc(out.float()).reshape(out.size(0), -1)
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


# turns a string into a list of each unique word with newline characters included
def tokenize(doc):

    punctuation_to_remove = string.punctuation.replace('\n', '')

    table = str.maketrans('', '', punctuation_to_remove)

    lines = doc.splitlines(keepends=True)

    tokens = []
    for line in lines:
        line = line.translate(table)
        words = line.split()
        tokens.extend(words)
        if line.endswith('\n'):
            tokens.append('\n')

    tokens = [token.lower() for token in tokens]

    return tokens


# uses the model and a seed string to generate lyrics
# AS OF RIGHT NOW IS NOT CONSIDERING ANY OF THE ATTRIBUTE VALUES
def generate(model, seed_str, attributes, word2int, word_array, len_generated_text=50, temperature=.8, top_p=0.8):

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
          _, hidden, cell = model(encoded_input[:, w].view(1), hidden, cell, attributes)

      last_word = encoded_input[:, -1]
    #   print(type(len_generated_text))
    #   print(range(len_generated_text))
      for i in range(150):
          logits, hidden, cell = model(last_word.view(1), hidden, cell, attributes)
          logits = torch.squeeze(logits, 0)
          last_word = top_p_sampling(logits, temperature, top_p)
          generated_str += " " + str(word_array[last_word])

    return generated_str.replace(" . ", ". ")


def parse_data():
    ### loads the dataset
    dataset = pandas.read_csv("songs_cleaned.csv")
    dataset['lyrics_y']

    ### removes the attributes we don't want to use, turns the ones we want into a tensor
    weird_attributes = ['world/life', 'shake the audience', 'family/gospel', 'communication', 'music', 'like/girls', 'feelings', 'loudness', 'instrumentalness', 'valence']
    dataset.drop(weird_attributes, axis=1)

    # turn the attributes that we want into a tensor
    good_attributes = ['dating', 'violence', 'romantic', 'obscene', 'sadness', 'danceability', 'energy', 'acousticness', 'night/time', 'movement/places', 'light/visual perceptions', 'family/spiritual']

    ### removes all punctuation from the lyrics (need to change this so that \n newline characters aren't removed too tho
    # set so that only words that aren't already in the set can get added
    unique_words = set()
    unique_words.add("\n")

    for lyrics in dataset['lyrics_y']:
        if isinstance(lyrics, str):

            # split the lyric string into individual word strings
            words = tokenize(lyrics)

            for word in words:
                unique_words.add(word.lower())

    vocab = sorted(unique_words) # vocab is the set of every unique word in the dataset

    ### gives every unique word a unique number that represents that word
    word2int = {word:i for i, word in enumerate(vocab)} # makes a dictionary with a corresponding number for each unique word
    word_array = np.array(vocab) # numpy array of all possible words

    # going to add these to the dataframe after looping through all the lyrics
    sequence_list = []
    target_list = []
    attribute_tensor_list = []

    ### now need to re-loop through the dataset, and turn each lyrics into the DataLoader object
    for index, row in dataset.iterrows():

        tokens = tokenize(row['lyrics_y'])

        # encodes the text from the lyrics as numbers
        text_encoded = np.array(
            [word2int[word] for word in tokens],
            dtype=np.int32
        )

        ### a sequence/chunk is used to make a list of previously seen words/the next word to come
        chunk_size = 9
        seq_length = 8
        # # not sure how to do this without batch size, so the batch size will be however many words are in that song
        # batch_size = len(tokens) 

        # makes chunks for all of the lyrics of however many words in a row
        text_chunks = [text_encoded[i:i+chunk_size] for i in range(len(text_encoded)-chunk_size+1)]

        sequences = torch.tensor(text_chunks[0:len(text_chunks) - 1])
        targets = torch.tensor(text_chunks[1:len(text_chunks)])

        # turning the text chunks into a dataset we can use (commented out lines are shuffling instances, which we don't want)
        sequence_list.append(sequences)
        target_list.append(targets)
        # indices = list(range(len(sequence_list)))
        # random.seed(12345)
        # random.shuffle(indices)
        # sequence_list = [sequence_list[i] for i in indices]
        # target_list = [target_list[i] for i in indices]
        
        # need to turn the attribute data into a tensor
        attribute_data = row[good_attributes]
        attributes = attribute_data.apply(pandas.to_numeric, errors='coerce') # fixing a type error in the next line
        attribute_data = torch.tensor(attributes)
        attribute_tensor_list.append(attribute_data)

        if index % 50 == 0:
            print(index)

    print(sequence_list[0])
    print(type(sequence_list[0]))

    dataset['seq_tensor'] = sequence_list
    dataset['target_tensor'] = target_list
    dataset['attribute_tensor'] = attribute_tensor_list

    print(dataset['seq_tensor'].iloc[0])
    print(type(dataset['seq_tensor'].iloc[0]))

    dataset.to_pickle('training_data.pkl')
    


def main():

    start_time = time.time()

    print('-------------------------------------------')
    print('Opening and parsing dataset')
    print('Time: ' + str(time.time() - start_time))
    print('-------------------------------------------')

    # parse_data()
    # exit()
    dataset = pandas.read_pickle("training_data.pkl")

    ### removes all punctuation from the lyrics (need to change this so that \n newline characters aren't removed too tho
    # set so that only words that aren't already in the set can get added
    unique_words = set()

    for lyrics in dataset['lyrics_y']:
        if isinstance(lyrics, str):

            # split the lyric string into individual word strings
            words = tokenize(lyrics)

            for word in words:
                unique_words.add(word.lower())

    vocab = sorted(unique_words) # vocab is the set of every unique word in the dataset

    ### gives every unique word a unique number that represents that word
    word2int = {word:i for i, word in enumerate(vocab)} # makes a dictionary with a corresponding number for each unique word
    word_array = np.array(vocab) # numpy array of all possible words

    ### creating the model

    print('-------------------------------------------')
    print('Creating model')
    print('Time: ' + str(time.time() - start_time))
    print('-------------------------------------------')

    # vocab size is used to create the size of the embedding layer
    vocab_size = len(word_array) # should be 39022
    # i THINK this is how many outputs the embedding layer has. So, the embedding layer maps however many unique words we have into 256 values
    embed_dim = 256
    rnn_hidden_size = 512 # THIS IS ACTUALLY THE LSTM LAYER SIZE AND THE HIDDEN LAYER SIZE
    chunk_size = 9
    seq_length = 8

    # initialize our model - using an LSTM layer, a hidden layer, and an output layer, passing in attributes before the output layer
    model = RNNfour(vocab_size, embed_dim, rnn_hidden_size, 12)

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
    # for epoch in range(num_epochs):
    #     hidden, cell = model.init_hidden(batch_size)
    #     seq_batch, target_batch = next(iter(seq_dl))
    #     seq_batch = seq_batch.to(DEVICE)
    #     target_batch = target_batch.to(DEVICE)
    #     optimizer.zero_grad()
    #     loss = 0 
    #     for w in range(seq_length):
    #         pred, hidden, cell = model(seq_batch[:, w], hidden, cell) # passing whichever batch in to the model with the hidden and cell state
    #         loss += loss_fn(pred, target_batch[:, w])
    #     loss.backward()
    #     optimizer.step()
    #     loss = loss.item()/seq_length
    #     if epoch % 10 == 0:
    #         print(f'Epoch {epoch} loss: {loss:.4f}')
    data_from_training = pandas.DataFrame(columns=["loss_per_100_songs", "epoch_training_time"])

    for epoch in range(100):
        epoch_start_time = time.time()
        epoch_loss = []
        for index, row in dataset.iterrows():
            seq_batch = row['seq_tensor']
            # print(seq_batch)
            target_batch = row['target_tensor']
            attributes = row['attribute_tensor']
            batch_size = len(seq_batch) # HOW MANY SEQ/TARGET PAIRS THERE ARE FOR THIS SONG
            hidden, cell = model.init_hidden(batch_size)
            seq_batch = seq_batch.to(DEVICE)
            target_batch = target_batch.to(DEVICE)
            attributes = attributes.to(DEVICE)
            optimizer.zero_grad()
            loss = 0 
            for w in range(seq_length):
                pred, hidden, cell = model(seq_batch[:, w], hidden, cell, attributes) # passing whichever batch in to the model with the hidden and cell state
                loss += loss_fn(pred, target_batch[:, w].to(torch.int64))
            loss.backward()
            optimizer.step()
            loss = loss.item()/seq_length
            if index % 100 == 0:
                print(f'Epoch {epoch + 1} song {index} loss: {loss:.4f}')
                epoch_loss.append(loss)


        if (epoch+1) in [1, 5, 10, 20, 50, 100]:
            filename = "rnn_model_" + str(epoch+1) + "_epoch.pth"
            torch.save(model, filename)

        training_time = epoch_start_time - time.time()
        # append this epochs data to the new dataframe
        data_from_training.loc[epoch] = [epoch_loss, training_time]
        # resave the dataframe to the csv file
        data_from_training.to_csv("data_from_training.csv")
    
    ### generate a new string
    generate(model, seed_str="I was")

if __name__ == "__main__":
    main()
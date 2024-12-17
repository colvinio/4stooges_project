import torch
import numpy as np
import string
from train_model import generate, RNNfour, tokenize
import torch
import numpy as np
import pandas

ordered_atributes = ['dating', 'violence',  'night/time', 'romantic', 'obscene', 'movement/places',
                    'light/visual perceptions', 'family/spiritual', 'sadness', 
                  'danceability', 'acousticness', 'energy']

attribute_list = ['dating', 'violence', 'romantic', 'obscene', 'sadness', 
                  'danceability', 'energy', 'acousticness', 'night/time', 
                  'movement/places', 'light/visual perceptions', 'family/spiritual']

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_user_attributes():
    """
    Prompt the user for each attribute value and return them as a torch.Tensor.
    """
    attribute_values = []
    for attr in attribute_list:
        val = input(f"Enter a value for {attr} (float): ")
        try:
            val = float(val)
        except ValueError:
            print(f"Invalid input for {attr}, defaulting to 0.0")
            val = 0.0
        attribute_values.append(val)
    #Convert the list of attributes to a tensor
    attribute_tensor = torch.tensor(attribute_values).float().to(DEVICE)
    return attribute_tensor, attribute_values

if __name__ == "__main__":
    
    
    ### NEED TO REOPEN THE ORIGINAL DATA TO CREATE SOME VARIABLES THAT WILL LET US GENERATE PREDICTIONS
    dataset = pandas.read_pickle("training_data.pkl")
    # print(dataset.columns)
    # exit()

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
    word2int = {word:i for i, word in enumerate(vocab)}
    word_array = np.array(vocab) # numpy array of all possible words
    ### DONE CREATING NECESSARY VARIABLES

    while 0 < 1:
        #Get the song name from the user
        song_name = input("Enter the name for the song (this will be the output filename): ")
        # song_name = 'test_obscene2'

        #Get attributes from the user
        # attributes, attribute_values = user_attributes = get_user_attributes()
        attribute_values = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        attributes = torch.tensor(attribute_values).float().to(DEVICE)
        attributes.to(DEVICE)

        #Load the model here
        # initialize our model - using an LSTM layer, a hidden layer, and an output layer, passing in attributes before the output layer
        # model = RNNfour(39022, 256, 512, 12)
        model = torch.load("rnn_model_100_epoch.pth")
        model.to(DEVICE)
        model.eval()

        seed_str = input("Enter a one word seed string for the lyrics generation (e.g. 'I'): ")
        if not seed_str.strip():
            seed_str = "I"

        # Generate lyrics
        generated_text = generate(model, seed_str, attributes, word2int, word_array)

        # Save to file
        output_filename = f"{song_name}.txt"
        with open(output_filename, "w", encoding="utf-8") as f:
            f.write(generated_text)
            f.write('\n')
            for i in range(len(attribute_list)):
                f.write('\n' + attribute_list[i] + ' = ' + str(attribute_values[i]))
            f.write('\nSeed string = ' + seed_str)

        print(f"Generated lyrics have been saved to {output_filename}")

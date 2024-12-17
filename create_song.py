import torch
import numpy as np
import string
from train_model import RNNfour, generate, word2int, word_array, DEVICE, tokenize, top_p_sampling
import torch
import numpy as np

attribute_list = ['dating', 'violence', 'romantic', 'obscene', 'sadness', 
                  'danceability', 'energy', 'acousticness', 'night/time', 
                  'movement/places', 'light/visual perceptions', 'family/spiritual']

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
    return attribute_tensor

if __name__ == "__main__":
    #Get the song name from the user
    song_name = input("Enter the name for the song (this will be the output filename): ")

    #Get attributes from the user
    user_attributes = get_user_attributes()

    #Load the model here
    model = torch.load("model_full.pth", map_location=DEVICE)
    model.to(DEVICE)
    model.eval()

    seed_str = input("Enter a one word seed string for the lyrics generation (e.g. 'I'): ")
    if not seed_str.strip():
        seed_str = "I"

    # Generate lyrics
    generated_text = generate(model, seed_str=seed_str, len_generated_text=50, temperature=.8, top_p=0.8)

    # Save to file
    output_filename = f"{song_name}.txt"
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(generated_text)

    print(f"Generated lyrics have been saved to {output_filename}")

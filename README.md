# 4stooges_project
CSCI373 Final Project Repo

-In the main file I left info about how we need to clean up the different files when we turn them into pandas df objects

Old dataset we looked at with attribute values and bad lyrics:
1950-2019: https://www.kaggle.com/datasets/saurabhshahane/music-dataset-1950-to-2019

2 new datasets with complete lyrics but not attribute values:
Song lyrics: https://www.kaggle.com/datasets/deepshah16/song-lyrics-dataset/data
Genius lyrics: https://www.kaggle.com/datasets/carlosgdcj/genius-song-lyrics-with-language-information

Website with good starting point for using an RNN to generate text: https://pytorch.org/tutorials/intermediate/char_rnn_generation_tutorial.html#nlp-from-scratch-generating-names-with-a-character-level-rnn

Website with tutorial on creating Text Generation using LSTM Networks (Character-based RNN): https://coderzcolumn.com/tutorials/artificial-intelligence/pytorch-text-generation-using-lstm-networks

Website with tutorial on creating lyric generator with RNN: https://www.activestate.com/blog/how-to-build-a-lyrics-generator-with-python-recurrent-neural-networks/



# To do:

In train_model.py I've moved all the code that was in tokenizing.ipynb so that it is more easily runnable. I've also commented everything so hopefully it's clear what is happening.

There are 4 RNN classes right now, each one is slightly different. RNNthree and RNNfour still need to get finished because I've not finished implementing the hidden layer.

The most important thing we need to do besides figure out the hidden layer is figure out how to use the attribute values when generating new lyrics.

Other things to do:
1. Change the tokenize() function so that it breaks a string up into individual words plus the newline character, instead of how it currently breaks strings into individual words and periods
2. When we parse the dataset we only want to keep certain attribute values. If you go to the email to adam with our check-in pdf, in the pdf there's a list of the 12 attribute values we want to use. We need to get rid of all the other attribute values from our instances

Here's the link to the guy talking about the code I'm reusing: https://www.youtube.com/watch?v=nzRIXaYAaqE&t=829s
There a link in the bio of the video to the website that he is scrolling through on the video if you want to take a look at that as well.
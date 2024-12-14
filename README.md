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

Here's the link to the guy talking about the code I'm reusing: https://www.youtube.com/watch?v=nzRIXaYAaqE&t=829s
There a link in the bio of the video to the website that he is scrolling through on the video if you want to take a look at that as well.


# To do:

In train_model.py I've moved all the code that was in tokenizing.ipynb so that it is more easily runnable. I've also commented everything so hopefully it's clear what is happening. There are 4 RNN classes right now, each one is slightly different. Depending on how long training takes we can test multiple of them, or we can just pick one of them.

The biggest thing left to do is reparsing the data in a different way so that the model can be trained.
After that, we have to actually train the model.
Depending on how long it takes to train the model once the implementation is correct, we can test the other models/hyperparameters. 

More important that creating a working model is having a complete report and presentation.

Other things to do:
1. Change the tokenize() function so that it breaks a string up into individual words plus the newline character, instead of how it currently breaks strings into individual words and periods
2. If we want a quantitative test of our model performance, figure that out

Figures we need to make for our final report/presentation:
1. Histograms of each of the 12 attributes we are using to show the ranges that they span.
2. Figure showing the architecture of our neural network models. This will probably have to be made in google drawings/manually, not coded
3. Some sort of figure to display our model outputs
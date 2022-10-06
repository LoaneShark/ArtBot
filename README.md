# ArtBot
### An bot that can read art

The ArtBot is a machine vision bot, trained on a dataset of nearly 80,000 paintings from the Kaggle 
competitions Painter By Numbers imageset. The final purpose of this bot is currently indeterminate. 

The bot uses a CNN-GAN (Convolutional Generative Adversarial Network) trained on the above imageset. The resulting model is capable of generate paintings, based on desired metadata flags (e.g. artist, style, genre, year). A `[WIP]` additional objective is to train a NLP model, most likely GPT-3, to also name the paintings accordingly.

The project was started in April 2018 by Bradley Beyers & Santiago Loane. It was built on Python 3.6 and
requires `numpy`, `scipy` and `sklearn`.

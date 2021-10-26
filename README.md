# Generating Music

An implementation of a bi-axial LSTM (Tensorflow 1) very similar to [Nikhil Kotecha's adaption](https://github.com/nikhil-kotecha/Generating_Music) (Tensorflow 1) of [Daniel D Johnson's bi-axial LSTM](https://github.com/danieldjohnson/biaxial-rnn-music-composition) (Theano) as described in  [this blog post](https://www.danieldjohnson.com/2015/08/03/composing-music-with-recurrent-neural-networks/).
In addition, the reinforcement learning part as described in Kotecha's thesis [Bach2Bach: Generating Music Using A Deep Reinforcement Learning Approach](https://arxiv.org/abs/1812.01060#) was attempted to be implemented by adapting the code (at that time) for the [RL_Tuner](https://github.com/magenta/magenta/tree/fbc059dbdd1c70071472e0b0707cb298f78ca9d2/magenta/models/rl_tuner), which is taken from the [Magenta](https://github.com/magenta/magenta) project.

## Requirements

First ensure that Python (version 3.8) is installed.

### Set up using pipenv

```
cd /path/to/this/project
pipenv install 
```


### Set up using pip

Install the dependencies listed in the Pipfile:

```
pip install matplotlib notebook jupyter numpy==1.15 tensor2tensor==1.3.0 pretty-midi==0.2.8 scipy==1.0.0 tensorflow==1.3
pip install git+https://github.com/louisabraham/python3-midi#egg=midi
```

## Directory structure 

```
generating-music-tf-2
│   README.md
│   LICENSE.md
│   Pipfile
|   Pipfile.lock   
│
└───data
│   │
│   └─── < composer_name >                      <- Contains all .mid files used
|                                                  for composer with composer_name
│   
└───notebooks
|   │   DL_modelling.ipynb                      <- Preprocessing and Training
|   │   DL_model_results_visualisation.ipynb    <- Results Analysis
|   │   DL_music_generation.ipynb               <- Generating new music
|   │   DRL_with_RL_Tuner.ipynb                 <- Deep Reinforcement Learning 
|   │                                              using the RL_Tuner
|   │   playground.ipynb                        <- Trying out different stuff
│   │
│   └───modules
|   |   |   
|   |   └───nikhil  <- Contains scripts with code copied and adapted from Kotecha
|   |   |   
|   |   └───magenta <- Contains scripts with code copied and adapted from Magenta                       
│   
└───outputs
│   │
│   └───midi 
|   |   |   
|   |   └───train                    <- .mid files generated during training
|   |   |   └─── < model_name >            
|   |   |   
|   |   └───generated                <- .mid files generated after training
|   |   |   └─── < model_name >             
|   |   
|   |   └───results                  <- Renamed and collected 
|   |   |   |                           .mid files based on data used for model
|   |   |   └─── < model_name >            
|   |   
│   └───models
|   |   |   
|   |   └───arrays                   <- numpy arrays saved after training
|   |   |   └─── < model_name >            
|   |   |   
|   |   └───ckpt                     <- saved models during and after training
|   |   |   └─── < model_name >         

```

## Using it

Run the ```notebook/DL_modelling.ipynb ```to train a model and save a model.
Load the saved model in ```notebook/DL_music_generation.ipynb ``` to generate new .mid files.
Run the ```notebook/DRL_with_RL_Tuner.ipynb ``` to train and generate music with the adapted RL_Tuner.
# deep-learning-image-colorization
Implementation of Colorful Image Colorization in Keras


## Set up folders
`mkdir training_images validation_images to_predict models`

Insert training and validation images in respective folders.


## Preprocessing of images
Create the hdf5 file for training and validation data by running

`python3 makeh5.py training && python3 makeh5.py validation`

## Training of model
Train the model by running 

`python3 network.py`

## Predicting
Images that you want the model to predict should be placed in the to_predict folder
Predict using the trained model by running

`python3 predictator.py model.h5`

## Please ⭐️ the repo if you liked it.
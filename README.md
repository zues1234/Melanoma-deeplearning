# Melanoma skin cancer detection

## Table of Content
  * [Overview](#overview)
  * [About](#About)
  * [Motivation](#motivation)
  * [Technical Aspect](#technical-aspect)
  * [Installation](#installation)
  * [To Do](#to-do)
  * [Technologies Used](#technologies-used)
  * [Credits](#credits)

## Overview
This is a simple melanoma detection application build using PyTorch and Flask. It uses a pretrained Serestnet50 model, with amost 33126 photos. The project was executed on Kaggle kernels.
This was a competition on kaggle [Melanoma](https://www.kaggle.com/c/siim-isic-melanoma-classification/overview)

## About
Skin cancer is the most prevalent type of cancer. Melanoma, specifically, is responsible for 75% of skin cancer deaths, despite being the least common skin cancer. 
The American Cancer Society estimates over 100,000 new melanoma cases will be diagnosed in 2020. It's also expected that almost 7,000 people will die from the disease. 
As with other cancers, early and accurate detection—potentially aided by data science—can make treatment more effective. be the last one you ever need -- I think this is it.

In this project, I identified melanoma in images of skin lesions. In particular, I've used images within the same patient and determine which are likely to represent a melanoma. 
Using patient-level contextual information may help the development of image analysis tools, which could better support clinical dermatologists.

Melanoma is a deadly disease, but if caught early, most melanomas can be cured with minor surgery. Image analysis tools that automate the diagnosis of melanoma will improve dermatologists' diagnostic accuracy. 
Better detection of melanoma has the opportunity to positively impact millions of people.

## Motivation
Cat & Dogs, autonomous vehicles, etc were all too mainstream and I wanted to work on something unique, and I came across this compeition [Melanoma](https://www.kaggle.com/c/siim-isic-melanoma-classification/overview) on kaggle and started working on it.

## Technical Aspect
This project is divided into two part:
1. Training a deep learning model using Pytoch.
      - model: SeResnet50 pretrained on imagenet
      - data: 33126 photos
      - evaluation metrics: ROC_AUC
      - special lib: wtfml [wtfml](https://github.com/abhishekkrthakur/wtfml)
2. Building a web app using Flask (will later host it on heroku too. )
   
## Installation
The Code is written in Python 3.8. If you don't have Python installed you can find it [here](https://www.python.org/downloads/). If you are using a lower version of Python you can upgrade using the pip package, ensuring you have the latest version of pip.
If you want to install the libraries & packages I've used, run the following command -- 

```sh
  pip install -r requirements.txt
  ```

## To Do
1. Convert the app to run without any internet connection, i.e. __PWA__.
2. Add a better vizualization chart to display the predictions.
3. deploy the Flask app on heroku with saved_model.pkl

## Bug / Feature Request
If you'd like to request a new function, feel free to do so by opening an issue [here](https://github.com/zues1234/Melanoma-deeplearning/issues/new). Please include sample queries and their corresponding results.

## Technologies used
  Major frameworks & Libraries 
  * [PyTorch](https://pytorch.org/)
  * [Flask](https://flask.palletsprojects.com/en/1.1.x/)
  * [Albumentation](https://albumentations.ai/)
  * [WTFML](https://pypi.org/project/wtfml/)
  * [Numpy | Pandas]






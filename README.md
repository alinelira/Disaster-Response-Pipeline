### Disaster Response Pipeline Project

## Table of Contents

1. [Project Overview](#overview)
2. [Files Description](#files)
3. [Instructions](#instructions)
4. [Acknowledgements](#licensing)

## Project Overview <a name="overview"></a>
The purpose of this project is to analyse real disaster data from [Appen](https://appen.com/) and build a machine learning pipeline, that uses NLTK and scikit-learn,  to output a model that classify messages sent during disasters into 36 categories (e.g. Medical Help, Medical Products, Offer, etc). 

There is also a web app that receives new messages as input and displays classification results in 36 predefined categories.
Below are some screenshots of the web app.
![image](https://user-images.githubusercontent.com/48845915/203915009-87df6ba6-1db1-40f6-a0f4-48a3c5389c07.png)
![image](https://user-images.githubusercontent.com/48845915/203915088-be4f19c8-1844-428e-b346-ab4ce1de1780.png)


## Files Description <a name="files"></a>

App folder:

    • 	Templates folder: Files to set-up the web app template
    •	run.py: File to run the web app
    
Data folder:

    •	DisasterResponse.db: Database that stores the cleaned dataset (df)
    •	disaster_categories.csv: Categories dataset
    •	disaster_messages.csv: Messages dataset
    •	ETL Pipeline Preparation.ipynb: ETL Preparation code
    •	process_data.py: ETL Pipeline
    
    
Models folder:

    •	classifier.pkl: Saved model
    •	ML Pipeline Preparation.ipynb: ML Preparation code
    •	train_classifier.py: Machine Learning Pipeline

    
README file

## Instructions <a name="instructions"></a>

1.	Run the following commands in the project's root directory to set up your database and model.
    
    •	Command to run ETL pipeline that cleans data and stores in database:
	
    	python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
	
    •	Command to run ML pipeline that trains classifier and saves model:
	
		python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl

2.	Run the following command in the app's directory to run your web app

    	python run.py
    
3.	Go to the web app that was generated in the previous step.


Note you should use Python versions 3.*. to run the codes above successfully. All required libraries (e.g. pandas, nltk, sklearn...) are avaialble in the [Anaconda distribution](https://www.anaconda.com/products/distribution)

## Acknowledgements <a name="licensing"></a>

I would like to thank Figure 8 for providing the datasets used to train the model, and also Udacity for all the training and guidance provided during this project. 

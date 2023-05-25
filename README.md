**Description**

This system uses Multilingual BERT to create embeddings for a code-mixed data 
set involving the languages Hindi and English.
The embeddings are then feeded to a LSTM to predict the sentiment of the 
sentences.

**Usage**
1. Run the setup.sh script
2. Run the train.sh script
3. Run the predict.sh script (optional)
4. Run the evaluate.sh script

**Scripts**

* baseline.py: creates a very simple baseline, where the sentiment with the highest frequency in the training data is predicted
* data_functions.py: procvides funtions for splitting the data and loading the data sets
* facebook_comments.py: provides functions to pre-process the data, so that it can be fed to Multlingual BERT
* model.py: provides functions, which are necessary for creating a model
* train.py: provides functions to create, train and save the model
* evaluate.py: loads the model and evaluates the accuracy and the fscore on the development and the test data
* main.py: the main function, which uses the provided functions to create, train and evaluate the model
* setup.sh: installs the requirements and sets up the environment
* train.sh: creates a model, trains it and saves the weights in outputs/
* predict.sh: makes predictions on the dev-set and the test-set and saves them in outputs/
* evaluate.sh: makes predictions on the dev-set and the test-set evaluates the predictions of the model

**our results**

| System | Classifier | Data | Train Accuracy | Test Accuracy |
| ------ | ------ | ------ | ------ | ------ |
| Uncased Bert | default | aclImdb_v1 | 0.973 | 0.919 |
| Multicase Bert | default | aclImdb_v1 | 0.968 | 0.889 |
| Multicase Bert | default | Hi-En | 0.825 | 0.644 |

**State of the Art results**

| Classifier | Data | Accuracy | F1-Score|
| ------ | ------ | ------ | ------ |
| CMSA | Hi-En | 0.835 | 0.827 |

**assigned paper results**

| Classifier | Data | Accuracy | F1-Score|
| ------ | ------ | ------ | ------ |
| Subword-LSTM | Hi-En | 0.697 | 0.658 |
| Subword-LSTM | SemEval' 13 | 0.606 | 0.537 |

**Baseline**

| Classifier | Data | Dev Accuracy | Test Accuracy |
| ------ | ------ | ------ | ------ |
| Bag Of Words | Hi-En | 0.443 | 0.395 |


#### Instructions for running the jupyternotebook
Requirements for this setup are:
* [JupyterLab](https://jupyter.org/install)
    * The [Jupytext](https://github.com/mwouts/jupytext#installation) plugin 
* (optional [Poetry](https://python-poetry.org/docs/#installation))
  
#### Setup

1. Install all [dependencies](pyproject.toml) in a virtual environment:
    * with poetry run `poetry install`
    * all dependencies are listed in [pyproject.toml](pyproject.toml)  
        make sure to include the dev-dependencies
    
1. Install the virtual environment as kernel for JupyterLab:
    * with poetry run `make install_kernel`
    * without poetry:
        1. activate your virtual environment
        1. run `python -m ipykernel install --user --name hi-en-sentiment --display-name "Python Hindi English Sentiment"`
        
1. Start Jupyterlab with `jupyter lab`

1. Open the [sentiment notebook](notebooks/sentiment.py) by rightclicking and selecting _Open With > Notebook_

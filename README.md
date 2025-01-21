# The Titanic project

## Introduction

Kaggle's **Titanic: Machine Learning from Disaster** is one of the most popular beginner-friendly projects for learning data science and machine learning. The goal of the project is to predict whether a passenger survived the Titanic shipwreck based on their characteristics, such as age, gender, ticket class, and more.

The raw pages from Kaggle are [here](https://www.kaggle.com/competitions/titanic/overview).

## Table of Contents

- .gitignore
- gender_submission.csv
- submission.csv
- test.csv
- train.csv
- test_data.json
- train_data.json
- process_data.py
- Training.py
- Trainingprocess.ipynb

### Explanations

The file `gender_submission.csv`, `test.csv` and `train.csv`are downloaded from the Kaggle website.

The file `submission.csv` is the final output result of the prediction, which has been committed to the Kaggle.

The `process_data.py` is used for transforming the `csv` files to `json` files, including `test_data.json` and `train_data.json`.

The `Trainingprocess.ipynb` is used for training the `json` data and getting the output.

The `Training.py` has the same content with `Trainingprocess.ipynb`.

## Implementation

The training process begins with **data cleaning and preprocessing**, which involves converting strings and filling in missing data. Next, **feature extraction** is performed on certain variables. After that, the **Random Forest algorithm** is used to train the model, and the prediction accuracy is evaluated while a **confusion matrix** is plotted. Finally, the trained model is applied to the **test set** to output the final results.

## Usage

The project can be an excellent experience to embark on the first machine learning project.

**Please ensure you have installed several packages below:**

```bash
pip install pandas
pip install numpy
pip install sklearn
```

**After the installation, you can train the model using commands below:**

```bash
git clone https://github.com/xiyuanyang-code/Titanic.git
cd Titanic
python process_data.py
python Training.py
```

Moreover, I suggest to run the `Trainingprocess.ipynb` in the Visual Studio Code, where you can run every cell independently to implement each function accordingly.

## Advertisement

[My personal Blog](https://xiyuanyang-code.github.io/posts/Python-advanced-File-Management/)

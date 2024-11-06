CSCI 680 AI for Software Engineering - If Statement
===

---

### Authors: 
1) Enze Xu (exu03@wm.edu)
2) Yi Lin (ylin13@wm.edu).
---

### 3-page Report:
[Assignment_Report_Enze_and_Yi.pdf](Assignment_Report_Enze_and_Yi.pdf)

---

### Result CSV Files:
These files are all available on [[OneDrive]](https://wmedu-my.sharepoint.com/:f:/g/personal/exu03_wm_edu/EgkSjqpbkqBAj1lqMXTdnF8BC3cHu9BN2DfHlh1yZquhng?e=xd8Lsd)
1) `provided-testset.csv`: The prediction result of `sample.csv`.

2) `generated-testset.csv`: The prediction result of `test_dataset.csv`.
---

### Quick screenshot of `generated-testset.csv`.
<img width="1249" alt="Screenshot 2024-11-06 at 07 45 40" src="https://github.com/user-attachments/assets/cbd79971-ca8c-4ac0-ac3d-37da37a0cc49">

---

### Quick Experimental Results

| **Dataset**      | **Result Save File**      | **Dataset Size** | **Correct Ratio** | Score 1 - avg | Score 1 - std | Score 2 - avg | Score 2 - std     | Score Avg - avg    | Score Avg - std     |
|------------------|---------------------------|------------------|--------------------|---------------|---------------|---------------|---------------|---------------|---------------|
| sample.csv       | provided-testset.csv      | 30               | 26.67%                                 | 93.98         | 4.89          | 63.22         | 27.72   | 78.60  | 15.99   |
| test_dataset.csv | generated-testset.csv     | 5,000            | 17.40%           | 92.88         | 5.94          | 60.22         | 27.73   | 76.55  | 15.96   |

---

# Contents

* [1 Introduction](#1-introduction)
* [2 Getting Started](#2-getting-started)
  * [2.1 Preparations](#21-preparations)
  * [2.2 Install Packages](#22-install-packages)
  * [2.3 Dataset and Data Preprocessing](#23-dataset-and-data-preprocessing)
  * [2.4 Run Pretraining](#24-run-pretraining)
  * [2.5 Run training](#25-run-training)
  * [2.5 Evaluate a CSV File Using the Trained Model Weight](#26-evaluate-a-csv-file-using-the-trained-model-weight)
* [3 Report](#3-report)
* [4 Questions](#4-questions)

[//]: # ()
[//]: # (---)

[//]: # (![heatmap]&#40;https://github.com/user-attachments/assets/4e0ca210-8325-46eb-b083-9740452cd5b4&#41;)

[//]: # ()
[//]: # ()
[//]: # (---)


# 1. Introduction

Training a Transformer model for Predicting if statements.

Your initial objective is creating two training datasets (i.e., pre-training and fine-tuning) required to train a Transformer model capable of automatically recommending suitable if statements in Python functions.


# 2. Getting Started


This project is developed using Python 3.9+ and is compatible with macOS or Linux


## 2.1 Preparations


(1) Clone the repository to your workspace.


```shell

~ $ git clone https://github.com/EnzeXu/CSCI680_If_Statement.git

```


(2) Navigate into the repository.

```shell

~ $ cd CSCI680_If_Statement

~/CSCI680_If_Statement $

```


(3) Create a new virtual environment and activate it. In this case we use Virtualenv environment (Here we assume you have installed the `virtualenv` package using you source python script), you can use other virtual environments instead (like conda).


For macOS or Linux operating systems:

```shell

~/CSCI680_If_Statement $ python -m venv ./venv/

~/CSCI680_If_Statement $ source venv/bin/activate

(venv) ~/CSCI680_If_Statement $ 

```


For Windows operating systems:


```shell

~/CSCI680_If_Statement $ python -m venv ./venv/

~/CSCI680_If_Statement $ .\venv\Scripts\activate

(venv) ~/CSCI680_If_Statement $ 

```


You can use the command deactivate to exit the virtual environment at any time.


## 2.2 Install Packages


```shell

(venv) ~/CSCI680_If_Statement $ pip install -r requirements.txt

```

## 2.3 Dataset and data preprocessing

You may download our full source dataset `full_dataset.csv` from [[OneDrive]](https://wmedu-my.sharepoint.com/:f:/g/personal/exu03_wm_edu/EgkSjqpbkqBAj1lqMXTdnF8BC3cHu9BN2DfHlh1yZquhng?e=xd8Lsd).

Please copy `full_dataset.csv` under directory `./src/`.

To preprocessing our dataset into the pretrain & train format:

```shell

(venv) ~/CSCI680_If_Statement $ python ifstatement/model/make_dataset_pretrain.py
(venv) ~/CSCI680_If_Statement $ python ifstatement/model/make_dataset_train.py

```

These steps will automatically read `./src/full_dataset.csv` and generate `./src/full_dataset_train.pkl` and `./src/full_dataset_pretrain.pkl`.

Alternatively, you may also directly download them via [[OneDrive]](https://wmedu-my.sharepoint.com/:f:/g/personal/exu03_wm_edu/EgkSjqpbkqBAj1lqMXTdnF8BC3cHu9BN2DfHlh1yZquhng?e=xd8Lsd).

In this way, please make sure they are copied to directory `./src/`.

## 2.4 Run Pretraining


```shell

(venv) ~/CSCI680_If_Statement $ python run_pretrain.py

```

## 2.5 Run training


```shell

(venv) ~/CSCI680_If_Statement $ python run_train.py

```

## 2.6 Evaluate a CSV File Using the Trained Model Weight

Please download the formatted cleaned test dataset `test_dataset.csv` and `sample.csv` from [[OneDrive]](https://wmedu-my.sharepoint.com/:f:/g/personal/exu03_wm_edu/EgkSjqpbkqBAj1lqMXTdnF8BC3cHu9BN2DfHlh1yZquhng?e=xd8Lsd).

Please save them under `./result/`.

The following script will read `./result/test_dataset.csv` and save the result csv into `./result/test_dataset_prediction.csv`.

```shell

(venv) ~/CSCI680_If_Statement $ python predict_dataset.py

```

You may also modify its input to `./result/sample.csv` in the main function.

# 3. Report


The assignment report is available in the attached file [Assignment_Report_Enze_and_Yi.pdf](Assignment_Report_Enze_and_Yi.pdf).



# 4. Questions


If you have any questions, please contact xezpku@gmail.com.




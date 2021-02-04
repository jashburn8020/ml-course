# Machine learning, data science, and deep learning with Python

## Setup

- Install Anaconda
  - run `conda init` during installation
    - adds some code into your `.bashrc` file
    - causes the default conda `base` environment to be activated with every shell session
- Relaunch command line terminal
- Disable automatic base environment activation: `conda config --set auto_activate_base False`
  - adds `auto_activate_base: false` to the `.condarc` file
- Clone the base environment into a new environment, `ml_course`: `conda create --name ml_course --clone base`
  - convenient because the base environment already has a number of packages preinstalled, e.g., `jupyter` and `matplotlib`
  - to list all conda environments: `conda info --envs`
  - to remove the environment: `conda remove --name myenv --all`
  - to manually create a new conda environment with Python 3.8: `conda create --name ml_course --python=3.8`
    - you will also need to install the following packages:
      - `jupyter`
      - `matplotlib`
      - `pandas`
      - `seaborn`
      - `scikit-learn`
      - `xlrd`
      - `statsmodels`
  - to install packages into a conda environment: `conda install [package [package ...]]`
  - to remove packages: `conda remove [package [package ...]]`
- Activate the new environment: `conda activate ml_course`
  - to deactivate the environment: `conda deactivate`
- Install PyDotPlus: `conda install pydotplus`
- Install TensorFlow: `conda install tensorflow`
- Download course material: <https://sundog-education.com/machine-learning/>
  - a collection of Jupyter Notebook files (`.ipynb`)
  - unzip the course material
- Test run Jupyter Notebook
  - in the course material directory: `jupyter notebook`
  - select one of the notebook files, e.g., `Outliers.ipynb`
- Alternatively, you can view and run Jupyter Notebook files in VS Code; see
  - <https://code.visualstudio.com/docs/python/jupyter-support>
  - <https://code.visualstudio.com/docs/python/data-science-tutorial>

## Introducing the Pandas Library

- See [PandasTutorial.ipynb](MLCourse/PandasTutorial.ipynb)

## Statistics and Probability Refresher

### Types of Data

- See [MLCourseSlides.pdf](MLCourseSlides.pdf), slides 5-10

### Using Mean, Median, and Mode in Python

- See:
  - [MeanMedianMode.ipynb](MLCourse/MeanMedianMode.ipynb)
  - [MeanMedianExercise.ipynb](MLCourse/MeanMedianExercise.ipynb)

### Standard Deviation and Variance

- See:
  - [MLCourseSlides.pdf](MLCourseSlides.pdf), slides 17-22
  - [StdDevVariance.ipynb](MLCourse/StdDevVariance.ipynb)

### Probability Density Function; Probability Mass Function

- Probability density function: probability of a range of values happening with continuous data
- Probability mass function: probabilities of given discrete values occurring in a data set
- See [MLCourseSlides.pdf](MLCourseSlides.pdf), slides 24-27

### Common Data Distributions (Normal, Binomial, Poisson, etc.)

- See [Distributions.ipynb](MLCourse/Distributions.ipynb)

### Percentiles and Moments

- See:
  - [MLCourseSlides.pdf](MLCourseSlides.pdf), slides 30-39
  - [Percentiles.ipynb](MLCourse/Percentiles.ipynb)
  - [Moments.ipynb](MLCourse/Moments.ipynb)

### A Crash Course in `matplotlib`

- See [MatPlotLib.ipynb](MLCourse/MatPlotLib.ipynb)

### Advanced Visualisation with Seaborn

- See [Seaborn.ipynb](MLCourse/Seaborn.ipynb)

### Covariance and Correlation

- See
  - [MLCourseSlides.pdf](MLCourseSlides.pdf), slides 41-45
  - [CovarianceCorrelation.ipynb](MLCourse/CovarianceCorrelation.ipynb)

### Conditional Probability

- See
  - [MLCourseSlides.pdf](MLCourseSlides.pdf), slides 48-49
  - [ConditionalProbabilityExercise.ipynb](MLCourse/ConditionalProbabilityExercise.ipynb)
  - [ConditionalProbabilitySolution.ipynb](MLCourse/ConditionalProbabilitySolution.ipynb)

### Bayes' Theorem

- See [MLCourseSlides.pdf](MLCourseSlides.pdf), slides 52-54

## Predictive Models

### Linear Regression

- See
  - [MLCourseSlides.pdf](MLCourseSlides.pdf), slides 56-62
  - [LinearRegression.ipynb](MLCourse/LinearRegression.ipynb)

### Polynomial Regression

- See
  - [MLCourseSlides.pdf](MLCourseSlides.pdf), slides 65-67
  - [PolynomialRegression.ipynb](MLCourse/PolynomialRegression.ipynb)

### Multiple Regression

- See
  - [MLCourseSlides.pdf](MLCourseSlides.pdf), slides 69-70
  - [MultipleRegression.ipynb](MLCourse/MultipleRegression.ipynb)

### Multi-Level Models

- See [MLCourseSlides.pdf](MLCourseSlides.pdf), slides 73-75

## Machine Learning with Python

### Supervised vs. Unsupervised Learning, and Train/Test

### Using Train/Test to Prevent Overfitting a Polynomial Regression

### Bayesian Methods: Concepts

### Implementing a Spam Classifier with Naive Bayes

### K-Means Clustering

### Clustering people based on income and age

### Measuring Entropy

### WINDOWS: Installing Graphviz

### MAC: Installing Graphviz

### LINUX: Installing Graphviz

### Decision Trees: Concepts

### Decision Trees: Predicting Hiring Decisions

### Ensemble Learning

### XGBoost

### Support Vector Machines (SVM) Overview

### Using SVM to cluster people using scikit-learn

## Recommender Systems

### User-Based Collaborative Filtering

### Item-Based Collaborative Filtering

### Finding Movie Similarities using Cosine Similarity

### Improving the Results of Movie Similarities

### Making Movie Recommendations with Item-Based Collaborative Filtering

### Improve the recommender's results

## More Data Mining and Machine Learning Techniques

### K-Nearest-Neighbors: Concepts

### Using KNN to predict a rating for a movie

### Dimensionality Reduction; Principal Component Analysis (PCA)

### PCA Example with the Iris data set

### Data Warehousing Overview: ETL and ELT

### Reinforcement Learning

### Reinforcement Learning & Q-Learning with Gym

### Understanding a Confusion Matrix

### Measuring Classifiers (Precision, Recall, F1, ROC, AUC)

## Dealing with Real-World Data

### Bias/Variance Tradeoff

### K-Fold Cross-Validation to avoid overfitting

### Data Cleaning and Normalization

### Cleaning web log data

### Normalizing numerical data

### Detecting outliers

### Feature Engineering and the Curse of Dimensionality

### Imputation Techniques for Missing Data

### Handling Unbalanced Data: Oversampling, Undersampling, and SMOTE

### Binning, Transforming, Encoding, Scaling, and Shuffling

## Apache Spark: Machine Learning on Big Data

### Warning about Java 11 and Spark 3!

### Spark installation notes for MacOS and Linux users

### Installing Spark - Part 1

### Installing Spark - Part 2

### Spark Introduction

### Spark and the Resilient Distributed Dataset (RDD)

### Introducing MLLib

### Introduction to Decision Trees in Spark

### K-Means Clustering in Spark

### TF / IDF

### Searching Wikipedia with Spark

### Using the Spark 2.0 DataFrame API for MLLib

## Experimental Design / ML in the Real World

### Deploying Models to Real-Time Systems

### A/B Testing Concepts

### T-Tests and P-Values

### Hands-on With T-Tests

### Determining How Long to Run an Experiment

### A/B Test Gotchas

## Deep Learning and Neural Networks

### Deep Learning Pre-Requisites

### The History of Artificial Neural Networks

### Deep Learning in the Tensorflow Playground

### Deep Learning Details

### Introducing Tensorflow

### Important note about Tensorflow 2

### Using Tensorflow, Part 1

### Using Tensorflow, Part 2

### Introducing Keras

### Using Keras to Predict Political Affiliations

### Convolutional Neural Networks (CNN's)

### Using CNN's for handwriting recognition

### Recurrent Neural Networks (RNN's)

### Using a RNN for sentiment analysis

### Transfer Learning

### Tuning Neural Networks: Learning Rate and Batch Size Hyperparameters

### Deep Learning Regularization with Dropout and Early Stopping

### The Ethics of Deep Learning

### Learning More about Deep Learning

## Final Project

### Your final project assignment: Mammogram Classification

### Final project review

## References

- <https://conda.io/projects/conda/en/latest/user-guide/getting-started.html>
- <https://docs.anaconda.com/anaconda/install/linux/>
- <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>
- <https://stackoverflow.com/a/58068850/13709997>
- <https://pandas.pydata.org/pandas-docs/stable/reference/index.html>
- <https://numpy.org/doc/stable/reference/index.html>
- <https://matplotlib.org/3.3.3/api/pyplot_summary.html>
- <https://docs.scipy.org/doc/scipy/reference/stats.html>
- <https://seaborn.pydata.org/index.html>

## Main Source

- Sundog Education. "Machine Learning, Data Science and Deep Learning with Python." _Udemy_, Udemy, September 2020, [www.udemy.com/course/data-science-and-machine-learning-with-python-hands-on/](https://www.udemy.com/course/data-science-and-machine-learning-with-python-hands-on/).

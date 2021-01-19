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

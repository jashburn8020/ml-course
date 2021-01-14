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
  - to install packages into a conda environment: `conda install [package [package ...]]`
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

## References

- <https://conda.io/projects/conda/en/latest/user-guide/getting-started.html>
- <https://docs.anaconda.com/anaconda/install/linux/>
- <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>
- <https://stackoverflow.com/a/58068850/13709997>

## Main Source

- Sundog Education. "Machine Learning, Data Science and Deep Learning with Python." _Udemy_, Udemy, September 2020, [www.udemy.com/course/data-science-and-machine-learning-with-python-hands-on/](https://www.udemy.com/course/data-science-and-machine-learning-with-python-hands-on/).

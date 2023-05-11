# Interpretable t-SNE Experiments

Code for reproducing the experiments found in our GLBIO oral presentation: "Towards Computing Attributions for Dimensionality Reduction Techniques"

---

Instructions
------------

To reproduce the results from the paper, you first want to install our package from TestPyPi:
    
    pip install -i https://test.pypi.org/simple/ interpretable-tsne

Alternatively, you can install from source using the following git repo:
    
    https://github.com/MattScicluna/interpretable_tsne

Dont forget to install the required packages:
    
    pip install -r requirements.txt

If you want to run the jupyter notebooks, then you will want to set up a kernel for your jupyter notebook:
    
    python -m ipykernel install --user --name <YOUR_ENV_NAME> --display-name "Python (<YOUR_ENV_NAME>)"

---

To run the experiments
----------------------

1) Download the datasets

    ```
    bash create_experiment_files.sh
    ```

    Note that all datasets (except for SARS-CoV-2) are available [here](https://drive.google.com/drive/folders/1dvoH6zYNIZNORoCuIdPo8GKDdmE4nBjs?usp=sharing)

2) Define paths in `set_environment_variables_public.sh`

3) Run experiments

    ```
    bash sim_data_experiment.sh
    bash mnist_experiment.sh
    bash mnist_create_table.sh
    bash sars_cov_2_experiment.sh
    ```

4) Check out the Notebooks to generate the figures found in the paper!

Tutorial
--------

We have a tutorial notebook available in `notebooks/synthetic_data_tutorial.ipynb`

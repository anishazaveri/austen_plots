# austen_plots
## Introduction
This repository contains demo data and code for "Sense and Sensitivity Analysis: Simple Post-Hoc
Analysis of Bias Due to Unobserved Confounding". 

## Requirements
The code has been tested on Python 3.7.3 with the following Python packages:  
scikit-learn==0.21.3  
scipy==1.2.1  
numpy==1.16.2  
pandas==0.25.1  
plotnine==0.6.0 (https://pypi.org/project/plotnine/)

## Instructions
To use this code (Use files under `example/` as reference)
1) Fit your data using any model and generate predictions for g, the propensity score, and Q, the conditional expected outcome.
2) Generate a .csv file with the following columns: 'g', 'Q', 't', 'y'. These correspond to the propensity score, the conditional expected outcome, the treatment and the outcome. For reference look at `input_df.csv` provided under `example/`
3) (Optional but recommended) Repeat step 1 with key covariates dropped before model fitting. For each such instance, generate a .csv file similar to step 2 but with column names modified to 'ghat', 'Qhat', 't', 'y'. Save all such files under a single folder (called `covariates` in the example)
4) Decide a menaningful amount of bias you would like to test for, based on domain knowledge about your dataset. Let's fix this as 2 for the example dataset
5) Run the following code (values correspond to the example dataset)  

`do_sensitivity.py -input_csv input_df.csv -bias -2 -output_dir sensitivity_output -covariate_dir covariates`

The output is an austen plot. Additionally, co-ordinates for the plot are provided as .csv files

By default `do_sensitivity.py` calculates bias for the ATE. If you would like to calculate bias for the ATT (i.e. restrict sensitivity analysis to datapoints in your `input.csv` file where t=1) add the flag `-do_att True` to the command above.


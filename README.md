# IITP_AGI_Causal Imitation Learning
---
This repository contains the implementation of an causal imitation learning model.

## Experiments
In this setup, S-piBD relies on a DAG estimated by the PC algorithm, and consequently its performance becomes sensitive to the quality of the causal discovery step.
To address this issue, we adapted and refined the causal discovery procedure to better suit our problem setting.
Using the example environment provided in Kumor et al., repeated experiments show that our modifications reduce the mean squared error (MSE) by approximately 11%.
## Usage
```
python imi_learn.py 10 500 5 
```
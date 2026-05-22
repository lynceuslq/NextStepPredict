This repo contains python modules of NextStepPredict, which contains catalysis prediction and enzyme design workflow, and example jupyter notebook to run the anslysis. 

# Workflow Information

The process is consisted of three steps including:
  1) training dataset preparation by single mutant selection and wet lab screening,
  2) machine-learning model training and validation, s
  3) everal rounds of protein mutation and mutant selection based on the ML model.
The process were engaged to select a combination of luciferase mutations that potentially maximise bioluminescence with a limited number of mutants evaluated by the wet lab.

First, the residues of single mutants for wet lab screening were selected from amino acid substitutions with more than 40% frequency in consensus sequences of the wild type luciferase. Those mutants were then tested for the log fold change (LFC) of bioluminescence values on the value of the wild type sequence. Embeddings of the single mutants were extracted by ESM-C model and filtered for features corresponding to the key residues involved in catalysis. Second, the feature-extracted embeddings of single mutants and their LFC values were combined to a dataset for the ML training step and split into 90% for training data and 10% for testing. An AdaBoost regressor with decision trees were used to train the dataset. Third, sequences with combinations of the mutations were generated and predicted by several rounds of mutations and LFC predictions. Each round of mutants was consisted by single mutations on the sequences with top accumulated LFC  from the former round, the mutants were estimated for LFC values by the ML model from step 2, which were added to the LFC values from all former mutations to get their accumulated LFC values. The mutants with the top 10 accumulated LFC values went to the next round of mutation and prediction, and the mutation pool for the next round was updated to the mutations in top 10 sequences at this round. After several rounds of mutations, mutation pool were used up, which ends the process. The proteins with combinations of mutations that accounted for top accumulated LFC values were then screened in the wet lab.

![prediction and generation](https://github.com/lynceuslq/NextStepPredict/blob/main/pred_and_generate_gpt.png)

# How to use the workflow
  1) train a prediction model with catalytic data from wetlab screening of single mutants and save it (example notebook: runpredictor.ipynb)
  2) generate sequences that combine the candidate mutations (example notebook: run_nextstepoptimizer.ipynb)


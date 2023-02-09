# Adversarial Perturbations for Robustness of Large Language Models
Final project for NLP 538 Fall 2022

We find that even large language models like BERT and GPT2 are susceptible to adversarial attacks. By using simple character and word-level perturbations, we were able to reduce the accuracy and the f1-score of these models on the Author Sentiment Prediction task.

We also tried to increase the amount of perturbation to see if we are able to get the model to re-classify examples correctly, to no avail.

However, through adversarial training, we see that these models can be made more robust to the aforementioned attacks. The only limitation to adversarial training is the inability to automate the generation of synthetic data. We need to constrain perturbations to prevent changing the original meaning of the input.

Python notebooks:
 - [BERT Sentiment Classifier training](https://github.com/nishitjain97/NLP_538_Fall_2022_Project_HaND/blob/main/BERT_Author_Sentiment_Classification.ipynb)
 - [GPT-2 Sentiment Classifier training](https://github.com/nishitjain97/NLP_538_Fall_2022_Project_HaND/blob/main/GPT_Author_Sentiment_Classification.ipynb)
 - [BERT prediction script](https://github.com/nishitjain97/NLP_538_Fall_2022_Project_HaND/blob/main/BERT_Predictions_and_Evaluation.ipynb)
 - [GPT-2 prediction script](https://github.com/nishitjain97/NLP_538_Fall_2022_Project_HaND/blob/main/GPT_Predictions_and_Evaluation.ipynb)
 
 [Fine-tuned Models](https://drive.google.com/drive/folders/1TsfgqDHbuQSC2oMwCiIV1TsU8L1uJaiJ?usp=share_link)
 
 [Datasets](https://drive.google.com/drive/folders/10t0Q9gAzWAzH57-3ikWX5D39o-zRjb5D?usp=share_link)

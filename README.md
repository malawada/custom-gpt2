# custom-gpt2
 gpt2 implementation and fine-tuning on custom datasets.

## Objective
The goal of this project was to build GPT-2 variants that could imitate the style of different corpuses (e.g., Wikipedia articles, Shakespeare) via finetuning. To achieve this goal, I used the Hugging Face APIs to load the pretrained model weights for GPT-2 and then trained the model for 3 additional epochs on each finetuning dataset. This resulted in a GPT-2 model that responds in the style of Wikipedia articles and a GPT-2 model that responds in the style of Shakespeare. 

## Files
main.ipynb: code for initial Hugging Face language model exploration. End of file analyzes how different generation strategies affect GPT-2 response quality.

train.py: code for training and evaluating GPT-2 models.

bert_sequence_classification.ipynb: code for exploring sequence classification tasks with Hugging Face.

x_training_log.txt: training console output from finetuning

sample_prompts_with_x.txt: samples of prompts and responses from original GPT-2 model and finetuned GPT2 model. X is the dataset used for finetuning.

## Installation
Primary dependencies are the Hugging Face libraries and PyTorch. All can be installed via requirements.txt


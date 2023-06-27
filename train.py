import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")


#identify candidate fine-tuning dataset

#implement fine-tuning loop and evaluation

#code for generating text from user prompts




if __name__ == '__main__':
    #load dataset
    #prepare data
    #train model
    #evaluate model
    #open user prompt loop
    pass
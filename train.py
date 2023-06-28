import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling, TrainingArguments, Trainer
from datasets import load_dataset
import pdb

#training parameters
checkpoint = 'distilgpt2'
dataset = 'tinyshakespeare'
epochs = 3

class GPTTrainer():

    def __init__(self, checkpoint):
        '''initialize tokenizer and model'''
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, cache_dir='E:\\cache')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(checkpoint, cache_dir='E:\\cache')
        self.data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)


    #identify candidate fine-tuning dataset
    def load_dataset(self, dataset):
        if dataset == 'wikitext':
            self.train_dataset = load_dataset(dataset, 'wikitext-2-v1', split='train', cache_dir='E:\\cache')
            self.val_dataset = load_dataset(dataset, 'wikitext-2-v1', split='validation', cache_dir='E:\\cache')
            self.train_dataset = self.train_dataset.filter(lambda x: len(x['text']) > 0) #remove empty rows
            self.val_dataset = self.val_dataset.filter(lambda x: len(x['text']) > 0) #remove empty rows
        elif dataset == 'tinyshakespeare':
            def chunk_text(x):
                chunks = []
                for item in x['text']:
                    chunks+=[item[i:i+256] for i in range (0, len(item), 256)]
                return {'text': chunks}
            dataset = load_dataset("tiny_shakespeare")
            self.train_dataset, self.val_dataset = dataset['train'], dataset['test']
            self.train_dataset = self.train_dataset.map(chunk_text, batched=True)
            self.val_dataset = self.val_dataset.map(chunk_text, batched=True)


    def prepare_data(self):
        '''tokenize dataset. apply batching, padding, splitting'''
        def tokenize_function(examples):
            return self.tokenizer(examples['text'], truncation=True, max_length=512)
        self.train_tokens = self.train_dataset.map(tokenize_function, batched=True)
        self.val_tokens = self.val_dataset.map(tokenize_function, batched=True)


    def finetune(self, epochs=1):
        '''implement fine-tuning loop and evaluation'''
        training_args = TrainingArguments('E:\\cache\\tinyshakespeare-finetuned-distilgpt2', 
                                          evaluation_strategy='steps', 
                                          eval_steps=500, 
                                          save_steps=500, 
                                          per_device_train_batch_size=4,
                                          per_device_eval_batch_size=4,
                                          num_train_epochs=epochs)
        trainer = Trainer(model=self.model,
                    args=training_args,
                    train_dataset=self.train_tokens,
                    eval_dataset=self.val_tokens,
                    data_collator=self.data_collator,
                    tokenizer=self.tokenizer)
        trainer.train()


    def generate(self, prompt, max_len):
        '''code for generating text from user prompts'''
        generated_text = self.model.generate(prompt, 
                                             max_length=max_len, 
                                             do_sample=True, 
                                             attention_mask=torch.ones(prompt.shape))
        generated_text = self.tokenizer.decode(generated_text[0], skip_special_tokens=True)
        return generated_text


    def prompt_loop(self):
        '''prompts user for input and generates text'''
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        while True:
            prompt = input('Enter a prompt: ')
            max_len = input("Max length: ")
            if max_len == "" or max_len == None:
                max_len = 50
            else:
                max_len = int(max_len)
            prompt = self.tokenizer.encode(prompt, return_tensors='pt')
            generated_text = self.generate(prompt, max_len=max_len)
            print(generated_text)


def multimodel_prompt_loop(original_trainer, finetuned_trainer):
    original_trainer.tokenizer.pad_token_id = original_trainer.tokenizer.eos_token_id
    finetuned_trainer.tokenizer.pad_token_id = finetuned_trainer.tokenizer.eos_token_id
    while True:
        prompt = input('Enter a prompt: ')
        max_len = input("Max length: ")
        if max_len == "" or max_len == None:
            max_len = 50
        else:
            max_len = int(max_len)
        orig_enc = original_trainer.tokenizer.encode(prompt, return_tensors='pt')
        finetuned_enc = finetuned_trainer.tokenizer.encode(prompt, return_tensors='pt')
        orig_gen = original_trainer.generate(orig_enc, max_len=max_len)
        finetuned_gen = finetuned_trainer.generate(finetuned_enc, max_len=max_len)
        print('Original model: ', orig_gen)
        print('Finetuned model: ', finetuned_gen)


if __name__ == '__main__':
    #finetuning model
    trainer = GPTTrainer(checkpoint)
    trainer.load_dataset(dataset)
    trainer.prepare_data()
    trainer.finetune(epochs)
    pdb.set_trace()

    del trainer

    #evaluating finetuned model vs original model
    original_trainer = GPTTrainer(checkpoint)
    finetuned_trainer = GPTTrainer('E:/cache/tinyshakespeare-finetuned-distilgpt2/checkpoint-2500')
    multimodel_prompt_loop(original_trainer, finetuned_trainer)
    # trainer.prompt_loop() #single model prompt loop

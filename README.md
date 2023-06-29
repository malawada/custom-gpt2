# custom-gpt2
 gpt2 implementation and fine-tuning on custom datasets.

## Objective
The goal of this project was to build GPT-2 variants that could imitate the style of different corpuses (e.g., Wikipedia articles, Shakespeare) via finetuning. To achieve this goal, I used the Hugging Face APIs to load the pretrained model weights for GPT-2 and then trained the model for 3 additional epochs on each finetuning dataset. This resulted in a GPT-2 model that responds in the style of Wikipedia articles and a GPT-2 model that responds in the style of Shakespeare. Total training time was 30-45 minutes per model on an Nvidia 2080S GPU with batch size 4.

## Files
main.ipynb: code for initial Hugging Face language model exploration. End of file analyzes how different generation strategies affect GPT-2 response quality.

train.py: code for training and evaluating GPT-2 models.

bert_sequence_classification.ipynb: code for exploring sequence classification tasks with Hugging Face.

x_training_log.txt: training console output from finetuning

sample_prompts_with_x.txt: samples of prompts and responses from original GPT-2 model and finetuned GPT2 model. X is the dataset used for finetuning.

## Installation
Primary dependencies are the Hugging Face libraries and PyTorch. All can be installed via requirements.txt

## Results
The output of GPT-2 varies significantly depending on the generation settings used. Using the default greedy token generation results in a large number of repetitions and circular phrases, while beam search can result in the model quickly going on tangents. From my experimentation in main.ipynb, I found that using multinomial sampling was the most effective way to get good, diverse responses with minimal repetitions. This setting was used for all models evaluated.

Below are some sample prompts and responses for each model variant.

### Original GPT-2 vs. TinyShakespeare Finetuned GPT-2
```
Enter a prompt: This shall not be the last time

Original model:  This shall not be the last time that an army of men will be assembled at Camp Hu'an-Ling for your sake," the military leaders said on Wednesday.
The United States imposed sanctions against several countries in response to allegations that the Iranian forces were taking steps to interfere with Iran's pursuit of nuclear and ballistic missiles in a bid to boost national security threats, the U.S. Foreign Office said in a December report. 
The American officials said it was an

Finetuned model:  This shall not be the last time that you see
Her husband's life upon account of this,
And that she be found; nor if 'tis true.'
EXETER:
Good lords.
KING RICHARD II:
He hath been very gentle with her,
Even I might have him in his household.
RUTLAND:
The one that she is for me is not by my heart;
And he, after the other, bears very
```

```
Enter a prompt: I bite my thumb at you

Original model:  I bite my thumb at you with his big hard-boiled, uncluttered fingers. He turns and says, "I have no way that I can not do this again, and I'll go."
He then hits his right hand lightly, forcing the other side of him on the ground. The other side of the room, the rest of the room, the body. Behind him he has the legs. He looks at the other side of his body, in the middle of

Finetuned model:  I bite my thumb at you. If it be so,
I know you will die, even as I am, to give it up.     
LADY ANNE:
Yes, no, but to the soul of this world.
LADY ANNE:
I fear you know you will die because you know not.    
LADY ANNE:
No.
LADY ANNE:
Heaven't said I'll kill me, or
```

```
Enter a prompt: The current united states economy is 

Original model:  The current united states economy is a great success but we should focus on the one-stop shop and not overwork and overconsumption of consumer goods. The more Americans who are unemployed and underused, the more we'll see the economy grow.”
The following chart has been taken from a report released by the American Enterprise Institute (EA), which has analyzed government spending and business investment by U.S. taxpayers last year. The report is intended to gauge how many Americans

Finetuned model:  The current united states economy is full of wars:
That of these, which they call the
Greatest Burgundy was the place of all the kingdoms
Against that which now lay them at liberty
To slaughter and pluck; for, on the other hand,
```


### Original GPT-2 vs. WikiText Finetuned GPT-2
```
Enter a prompt: In his early life, he was a

Original model:  In his early life, he was a little boy who used music to find his way to school without any problem, and after he was diagnosed he developed a severe asthma. He was diagnosed with bronchitis, and he died a year later.

Finetuned model:  In his early life, he was a member of the Church of Scientology and <unk>. In 1985, he received a <unk> from several friends who 
wished to help him develop a religious identity. Soon after graduating from the university in 1986,
```

```
Enter a prompt: Over fifty percent of 

Original model:  Over fifty percent of the United States had been hit during his first eight years in office — a major gain for Republican presidential contenders. During his second term, Mr. Trump's approval ratings fell by only one point, from an unfavorable rating of 26 percent

Finetuned model:  Over fifty percent of all non @-@ migrant workers are employed in the restaurant industry, with nearly seven percent employed by 
the restaurant industry. <unk> % are migrant <unk>, but 80 percent are employed from private companies. The food chain
```

```
Enter a prompt: This year's superbowl was controversial because

Original model:  This year's superbowl was controversial because there was no one to stand in front of it. A great majority of the birds had gone for the wild, but their wild status was threatened by habitat loss.

Finetuned model:  This year's superbowl was controversial because of possible competition from the New Zealand Institute for Conservation. A group 
led by Sir Patrick <unk> described the species as " very <unk> " in a review of The New Zealand Times, saying the species
```
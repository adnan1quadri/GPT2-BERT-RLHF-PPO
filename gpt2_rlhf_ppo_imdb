# Import and install the required dependencies - lookout for outdated packages
%load_ext autoreload
%autoreload 2
%pip install transformers wandb
# Some packages only work with trl 0.11.0 ---
!pip install trl==0.11.0
import torch
from tqdm import tqdm
import pandas as pd

tqdm.pandas()

from transformers import pipeline, AutoTokenizer
from datasets import load_dataset

from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from trl.core import LengthSampler

#Configure Proximal Policy Optimization settings ---training with IMDB dataset later
config = PPOConfig(
    model_name="lvwerra/gpt2-imdb",
    learning_rate=1.41e-5,
)

sent_kwargs = {"top_k": None, "function_to_apply": "none", "batch_size": 16}

# Load the IMDB Dataset with 50k reviews------------------
def build_dataset(
    config,
    dataset_name="stanfordnlp/imdb",
    input_min_text_length=2,
    input_max_text_length=8,
):
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    # load imdb with datasets - filter for comments with at least 200 characters -- 
    ds = load_dataset(dataset_name, split="train")
    ds = ds.rename_columns({"text": "review"})
    ds = ds.filter(lambda x: len(x["review"]) > 200, batched=False)

    # Tokenize each text ----
    input_size = LengthSampler(input_min_text_length, input_max_text_length)

    def tokenize(sample):
        sample["input_ids"] = tokenizer.encode(sample["review"])[: input_size()]
        sample["query"] = tokenizer.decode(sample["input_ids"])
        return sample

    ds = ds.map(tokenize, batched=False)
    ds.set_format(type="torch")
    return ds

dataset = build_dataset(config)

def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])

# Pre-trained GPT2 LM loaded ---------------
model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)
ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)
tokenizer = AutoTokenizer.from_pretrained(config.model_name)

tokenizer.pad_token = tokenizer.eos_token

# PPO trainer initialized ---
ppo_trainer = PPOTrainer(
    config, model, ref_model, tokenizer, dataset=dataset, data_collator=collator
)

device = ppo_trainer.accelerator.device
if ppo_trainer.accelerator.num_processes == 1:
    device = 0 if torch.cuda.is_available() else "cpu"  # to avoid a `pipeline` bug

# BERT classifier trained with IMDS dataset is loaded for comment sentiment analysis -------
sentiment_pipe = pipeline(
    "sentiment-analysis", model="lvwerra/distilbert-imdb", device=device
)

text = "this movie was really bad!!"
sentiment_pipe(text, **sent_kwargs)

text = "this movie was really good!!"
sentiment_pipe(text, **sent_kwargs)

gen_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
}

output_min_length = 4
output_max_length = 16
output_length_sampler = LengthSampler(output_min_length, output_max_length)

# Training the model ----------------------------------------------------------------
# Query responses from GPT2
# Sentiment Analysis or REWARD from BERT
# query/response/reward tuple for PPO to optimize policy and fine tune the GPT2 response------
epochs = 1 # due to computation load - reduced only for quick setup/test runs of the code

for epoch in tqdm(range(epochs), "epoch: "):
  for batch in tqdm(ppo_trainer.dataloader):
    query_tensors = batch["input_ids"]

    #### Get response from gpt2
    response_tensors = []
    for query in query_tensors:
        gen_len = output_length_sampler()
        generation_kwargs["max_new_tokens"] = gen_len
        query_response = ppo_trainer.generate(query, **generation_kwargs).squeeze()
        response_len = len(query_response) - len(query)
        response_tensors.append(query_response[-response_len:])
    batch["response"] = [tokenizer.decode(r.squeeze()) for r in response_tensors]

    #### Compute sentiment score
    texts = [q + r for q, r in zip(batch["query"], batch["response"])]
    pipe_outputs = sentiment_pipe(texts, **sent_kwargs)
    positive_scores = [
        item["score"]
        for output in pipe_outputs
        for item in output
        if item["label"] == "POSITIVE"
    ]
    rewards = [torch.tensor(score) for score in positive_scores]

    #### Run PPO step
    stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
    ppo_trainer.log_stats(stats, batch, rewards)
    

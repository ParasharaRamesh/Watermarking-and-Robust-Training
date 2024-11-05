import sys
import torch
from transformers import AutoTokenizer
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList
from transformers import GPT2LMHeadModel
import random
import numpy as np


# Watermarking scheme: modify sampling strategy
# Generates random number for every word iteration
class MyWatermarkLogitsProcessor(LogitsProcessor):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        # r = a random value between 0 and 1
        r = random.random()

        # scores_processed = probabilities of each word, summing up to 1, shape is (1, V) where V is size of vocabulary
        scores_processed = scores.clone().softmax(dim=-1).to(torch.float64)

        # select the word using r (currently selecting the first word in vocab)

        # find out the cummulative probability scores (across the last dimension) by creating V buckets where each bucket corresponds to each word in the vocabulary. By finding out which bucket the 'r' falls in we can sample that word
        bucket_probs = scores_processed.cumsum(dim=-1).to(torch.float64)[0]
        next_token_id = torch.searchsorted(bucket_probs, r).item()

        # Change score of next_token to inf, and scores of all other words to -inf
        # Forcing the model to choose next_token
        vocab_tensor = torch.arange(scores.shape[-1], device=scores.device)
        next_token_mask = torch.isin(vocab_tensor, next_token_id)  # boolean mask for each of the elements

        scores_processed = scores.masked_fill(next_token_mask, float(
            "inf"))  # only the element at next_token_id will be made inf while the rest remains as is

        scores_processed = scores_processed.masked_fill(~next_token_mask, -float(
            "inf"))  # all other entries apart from the next token is made as -inf
        return scores_processed


# Model watermarked with a SECRET_KEY
class MyWatermarkedModel(GPT2LMHeadModel):
    def __init__(self, config, sk: int, **kwargs):
        super().__init__(config, **kwargs)
        self.__sk = sk
        random.seed(self.__sk)

    # Reset seed with secret key for reproduced generations
    def reset_seed(self):
        print("Reset seed")
        random.seed(self.__sk)

    def generate(
            self,
            **kwargs,
    ):
        logits_processor = LogitsProcessorList([MyWatermarkLogitsProcessor()])
        outputs = super().generate(logits_processor=logits_processor, **kwargs)
        orig_outputs = outputs

        # Compute original scores
        output_ids = outputs.sequences
        input_ids = kwargs.pop('input_ids')
        attention_mask = kwargs.pop('attention_mask')
        input_len = input_ids.shape[-1]
        output_len = output_ids.shape[-1]
        scores = []
        for i in range(input_len, output_len):
            input_ids = output_ids[:, :i]
            attention_mask = torch.full(input_ids.size(), 1)
            outputs = super().generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=1,
                                       return_dict_in_generate=True, output_scores=True)
            scores.append(outputs.scores[0])
        outputs.scores = tuple(scores)
        outputs.sequences = output_ids

        return outputs


def query_model(input_str, model, tokenizer, max_new_tokens):
    inputs = tokenizer(input_str, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, return_dict_in_generate=True,
                             output_scores=True)
    output_str = [tokenizer.decode(x) for x in outputs.sequences][0]
    return output_str


def verify_str(input_str, sk, model, tokenizer, max_new_tokens, is_model_watermarked=False):
    # Generate list of r values
    random.seed(sk)
    rs = [random.random() for _ in range(max_new_tokens)]
    if is_model_watermarked:
        # Generate tokens with watermarked model
        model.reset_seed()
    else:
        # if the model is not watermarked it does not have the reset_seed function
        random.seed(sk)
    inputs = tokenizer(input_str, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, return_dict_in_generate=True,
                             output_scores=True)
    input_len = inputs['input_ids'].shape[-1]
    valids = []
    for i in range(max_new_tokens):
        # Get probability of i-th word
        next_token_id = outputs.sequences[0][input_len + i]
        next_token = tokenizer.decode(next_token_id)
        scores = outputs.scores[i]
        scores_processed = scores.clone().flatten().softmax(dim=-1)
        r = rs[i]

        # Using same logic as finding out the cummulative probability buckets to find out the next token id
        bucket_probs = scores_processed.cumsum(dim=-1)
        check_next_token_id = torch.searchsorted(bucket_probs, r)

        # check if the tokens are indeed the same
        valid = check_next_token_id.item() == next_token_id.item()

        valids.append(valid)

    # Check if 90% of generated tokens pass our verifier check
    if np.array(valids).mean() >= 0.9:
        return True
    else:
        return False


# Verify if the model is watermarked with a given SECRET_KEY
def verifier(sk, model, tokenizer, max_new_tokens):
    input_str = "Hello, my name is"
    if verify_str(input_str, sk, model, tokenizer, max_new_tokens) == True:
        print("Given model IS watermarked with the given secret key")
    else:
        print("Given model is NOT watermarked with the given secret key")


if __name__ == '__main__':
    MAX_NEW_TOKENS = 10
    SECRET_KEY = random.randrange(sys.maxsize)

    tokenizer = AutoTokenizer.from_pretrained("distilbert/distilgpt2")

    MODEL_ORIG = GPT2LMHeadModel.from_pretrained("distilbert/distilgpt2")
    MODEL_ORIG.generation_config.pad_token_id = tokenizer.eos_token_id
    model = MyWatermarkedModel.from_pretrained("distilbert/distilgpt2", sk=SECRET_KEY)
    model.generation_config.pad_token_id = tokenizer.eos_token_id

    prompts = [
        "Hello, my dog is cute",
        "Good morning, my"
    ]

    for i, input_str in enumerate(prompts):
        print(f"{i + 1}. Input str is `{input_str}`")

        print(f" -> Original model outputs..")
        print(query_model(input_str, MODEL_ORIG, tokenizer, max_new_tokens=MAX_NEW_TOKENS))
        print("Verifying if output is watermarked.. (Should be False)")
        print(verify_str(input_str, SECRET_KEY, MODEL_ORIG, tokenizer, max_new_tokens=MAX_NEW_TOKENS))
        print("-" * 50)

        print(f" -> New watermarked model outputs..")
        print(query_model(input_str, model, tokenizer, max_new_tokens=MAX_NEW_TOKENS))
        print("Verifying if output is watermarked.. (Should be True)")
        print(verify_str(input_str, SECRET_KEY, model, tokenizer, max_new_tokens=MAX_NEW_TOKENS,
                         is_model_watermarked=True))
        print("-" * 50)

        print("=" * 50)
    print()

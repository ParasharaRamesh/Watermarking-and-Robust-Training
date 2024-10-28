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
        # scores_processed = probabilities of each word, summing up to 1
        scores_processed = scores.clone().softmax(dim=-1)

        # TODO: select the word using r (currently selecting the first word in vocab)
        next_token_id = 0

        # Change score of next_token to inf, and scores of all other words to -inf
        # Forcing the model to choose next_token
        vocab_tensor = torch.arange(scores.shape[-1], device=scores.device)
        next_token_mask = torch.isin(vocab_tensor, next_token_id)
        scores_processed = scores.masked_fill(next_token_mask, float("inf"))
        scores_processed = scores.masked_fill(~next_token_mask, -float("inf"))
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
        scores=[]
        for i in range(input_len, output_len):
            input_ids = output_ids[:, :i]
            attention_mask = torch.full(input_ids.size(), 1)
            outputs = super().generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=1, return_dict_in_generate=True,output_scores=True)
            scores.append(outputs.scores[0])
        outputs.scores = tuple(scores)
        
        return outputs

def query_model(input_str, model, tokenizer, max_new_tokens):
    inputs = tokenizer(input_str, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, return_dict_in_generate=True,
                                output_scores=True)
    output_str = [tokenizer.decode(x) for x in outputs.sequences][0]
    return output_str


def verify_str(input_str, sk, model, tokenizer, max_new_tokens):
    # Generate list of r values
    random.seed(sk)
    rs = [random.random() for _ in range(max_new_tokens)]
    # Generate tokens with model
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
        # TODO: Check if the next token is the one chosen by r
        valid=True
        
        valids.append(valid)
    # Check if 90% of generated tokens pass our verifier check
    if np.array(valids).mean() >= 0.9:
        return True
    else:
        return False

# Verify if the model is watermarked with a given SECRET_KEY
def verifier(sk, model, tokenizer, max_new_tokens):
    input_str="Hello, my name is"
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

    prompts=[
        "Hello, my dog is cute",
        "Good morning, my"
    ]

    for input_str in prompts:
        print(query_model(input_str, MODEL_ORIG, tokenizer, max_new_tokens=MAX_NEW_TOKENS))
        print(query_model(input_str, model, tokenizer, max_new_tokens=MAX_NEW_TOKENS))
        # print(verify_str(input_str, SECRET_KEY, model, tokenizer, max_new_tokens=MAX_NEW_TOKENS))



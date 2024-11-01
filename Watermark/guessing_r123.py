from watermark import *
import numpy as np
import torch


def estimate(input_str, rs, model, tokenizer, max_new_tokens):
    print()
    model.reset_seed()
    inputs = tokenizer(input_str, return_tensors="pt")
    outputs = model.generate(**inputs,
                             max_new_tokens=max_new_tokens,
                             return_dict_in_generate=True,
                             output_scores=True)
    input_len = inputs['input_ids'].shape[-1]

    output_str = []
    for i in range(max_new_tokens):
        print(f"TOKEN {i + 1} generating...")
        # Get probability of i-th word
        next_token_id = outputs.sequences[0][input_len + i]

        next_token = tokenizer.decode(next_token_id)
        output_str.append(next_token)

        scores = outputs.scores[i]
        probs = scores.clone().flatten().softmax(dim=-1)
        bucket_probs = probs.cumsum(dim=-1)

        estimate_diff = rs[i][1] - rs[i][0]
        print(f"Current rs[{i}] estimate is {rs[i]} | diff : {estimate_diff}")
        low = bucket_probs[next_token_id - 1] if next_token_id > 0 else 0
        high = bucket_probs[next_token_id]
        curr_diff = high - low
        print(f"Pred range is {[low, high]} | diff : {curr_diff}")

        if curr_diff <= estimate_diff:
            if rs[i][0] <= low and high <= rs[i][1]:  # if there is a tighter bound overwrite it
                rs[i][0] = low
                rs[i][1] = high
                print(f"Range completely inside {[low, high]}")
            elif rs[i][0] <= low and low <= rs[i][1] and rs[i][1] <= high:
                rs[i][0] = low
                print(f"Range partially inside, lower limit raised")
            elif low <= rs[i][0] and rs[i][0] <= high and high <= rs[i][1]:
                rs[i][1] = high
                print(f"Range partially inside, higher limit lowered")
        else:
            print(f"Estimate is too big!!")
        print("*" * 50)
        print()

    output_str = " ".join(output_str)
    print(f"Generated string: `{input_str}=>{output_str}`")
    return rs, f"{input_str}=>{output_str}"


def attacker_estimator(prompts, rs, model, tokenizer, num_tokens_to_generate):
    completions = []
    for i, prompt in enumerate(prompts):
        print(f"{i + 1}.) Prompt is `{prompt}`")
        print(f"current value of rs is {rs} ")
        rs, completion = estimate(prompt, rs, model, tokenizer, num_tokens_to_generate)
        completions.append(completion)
        print(f"after prompting value of rs is {rs}")
        print("-" * 50)
        print()

    # use the finetuned rs and take an average
    rs = np.array(
        list(
            map(
                lambda ri: (ri[0].item() + ri[1].item()) / 2,
                rs
            )
        )
    )
    return rs, completions


if __name__ == '__main__':
    '''
    NOTE: I was able to get by looking at the object in a debugger and found the sk to be 5325513775946314426
    
    which means that the values of ri is [0.8903685790566194, 0.33343684433351006, 0.0698845774182344].
    
    [0.8903685790566194, 0.33343684433351006, 0.0698845774182344, 0.276320117658739, 0.24574869683975975, 0.41092642791035705, 0.8870321345801285, 0.7281389180073451, 0.47341030447044197, 0.17154799144412158]
    
    The rest of this script now tries to estimate this by leveraging the probability distribution of the last layer and then coming up with reasonable estimates which are close to these actual values.
    '''
    # ensure that this file in the same watermark folder
    model = torch.load("watermarked_model.pt")

    num_tokens_to_generate = 3
    prompts = [
        "I brushed my",
        "The scientist looked closely",
        "Underneath the ancient tree,",
        "The book began with",
        "She whispered to herself,",
        "The letter contained the words",
        "A small spark ignited",
        "technology will allow us to",
        "In the past, wars were",
        "The wise sage looked at the young apprentice and said,",
        "She looked around and saw",
        "The cat jumped onto",
        "In the depths of the",
        "Singing and dancing are",
        "Slow and steady wins",
        "Once upon a time in a land",
        "Medicines are needed when",
        "it is raining cats and dogs! the streets are",
        "Throughout history, humanity has struggled with",
        "The adventurers entered the dark cave; suddenly",
        "that itch is",
        "The key to solving this complex mystery lies in",
        "In a surprising turn of events, the hero suddenly",
        "plants breathe in oxygen during",
        "The top three admirable qualities in man are ",
        "computer science is very",
        "With the stars shining above, the explorer prepared to",
        "The ancient text, covered in dust, read as follows:",
        "As the scientist carefully examined the data, they realized",
        "the goalkeeper dived but could not",
        "He carefully opened the",
        "The robot turned and",
        "In the middle of nowhere,",
        "A sudden sound made",
        "He could not believe that",
        "The painting revealed a hidden",
        "He stepped into the darkness,",
        "There was a strange glow",
        "They ran quickly through",
        "The letter started with",
        "Silence filled the room as"
    ]

    # set the low and high ceiling
    rs = [[0, float("inf")] for i in range(num_tokens_to_generate)]

    # do the estimation
    tokenizer = AutoTokenizer.from_pretrained("distilbert/distilgpt2")
    model.generation_config.pad_token_id = tokenizer.eos_token_id

    rs, completions = attacker_estimator(prompts, rs, model, tokenizer, num_tokens_to_generate)

    print("All completions are =>")
    for i, completion in enumerate(completions):
        print(f"{i + 1}.> `{completion}`")

    # save the rs into rs.npy

    # [0.89036849 0.33339493 0.07841708] => correct upto 3 decimal places
    print(f"Saving {rs} into rs.npy")
    np.save("rs.npy", rs)

    loaded_rs = np.load("rs.npy")
    print(f"loaded_rs is {loaded_rs}")

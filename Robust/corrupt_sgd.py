import torch
from torch import nn
from tqdm import tqdm
from torch.optim import AdamW
from transformers import BertForSequenceClassification, BertTokenizer
import argparse
import random
import codecs
import numpy as np
import os
from robust_aggregator import robust_aggregator
import logging
import autograd_hacks

# Compute accuracy
def binary_accuracy(preds, y):
    rounded_preds = torch.argmax(preds, dim=1)
    correct = (rounded_preds == y).float()
    acc_num = correct.sum()
    acc = acc_num / len(correct)
    return acc_num, acc

def process_data(data_file_path, seed):
    logger.info("Loading file " + data_file_path)
    random.seed(seed)
    all_data = codecs.open(data_file_path, 'r', 'utf-8').read().strip().split('\n')[1:]
    random.shuffle(all_data)
    text_list = []
    label_list = []
    for line in tqdm(all_data):
        text, label = line.split('\t')
        text_list.append(text.strip())
        label_list.append(float(label.strip()))
    return text_list, label_list

def process_model(model_path, device):
    logger.info("Loading model " + model_path)
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path, return_dict=True)
    model = model.to(device)
    parallel_model = nn.DataParallel(model)
    return model, parallel_model, tokenizer

# Randomly sample epsilon fraction of the gradients to corrupt
def corrupt_gradients(stacked_grads, epsilon):
    """
    Corrupts epsilon fraction of individual gradients
    Args:
        stacked_grads (tensor): Individual gradients for each data point, size = (batch_size, 2*768)
        epsilon (float): Fraction of gradients to corrupt (0 < epsilon <= 1).
    Returns:
        corrupted_gradients: List of gradients after applying corruption.
    """
    # TODO: corrupt epsilon fraction of the batch of gradients
    
    return stacked_grads

# Generic train procedure for single batch of data
# Simulate corrupting epsilon fraction of gradients of the classification layer
def corrupt_train_iter(model, batch, labels, optimizer, criterion, epsilon):
    outputs = model(**batch)
    acc_num, acc = binary_accuracy(outputs.logits, labels)
    loss = criterion(outputs.logits, labels)
    loss.backward(retain_graph=True)
    # Get individual gradients of classifier layer weights, for each data sample in the batch
    last_layer = dict(model.named_modules())['classifier']
    autograd_hacks.compute_grad1(model)
    stacked_grads = last_layer.weight.grad1
    autograd_hacks.clear_backprops(model)
    model.zero_grad()

    orig_cnt = stacked_grads.shape[0]
    stacked_grads = stacked_grads.view(orig_cnt, -1)

    # Randomly sample epsilon fraction of the gradients to corrupt
    stacked_grads = corrupt_gradients(stacked_grads, epsilon)

    # TODO: Import your implementation of robust aggregator in Question 1
    stacked_grads = robust_aggregator(stacked_grads)
    filtered_cnt = orig_cnt - stacked_grads.shape[0]
    logger.info(f"Acc {acc_num}, Filtered out {filtered_cnt} gradients")

    # Compute average on corrupted gradients    
    last_grad = torch.mean(stacked_grads, dim=0)
    last_grad = last_grad.view(2, 768)
    last_layer.weight.grad = last_grad
    optimizer.step()
    optimizer.zero_grad()
    return loss, acc_num, filtered_cnt

# Generic train function for single epoch (over all batches of data)
def corrupt_train_epoch(model, tokenizer, train_text_list, train_label_list,
                batch_size, optimizer, criterion, device, epsilon):
    """
    Generic train function for single epoch (over all batches of data)

    Parameters
    ----------
    model: model to be attacked
    tokenizer: tokenizer
    train_text_list: list of training set texts
    train_label_list: list of training set labels
    optimizer: Adam optimizer
    criterion: loss function
    device: cpu or gpu device
    epsilon: fraction of gradients to corrupt (0 < epsilon <= 1).

    Returns
    -------
    updated model
    average loss over training data
    average accuracy over training data

    """
    epoch_loss = 0
    epoch_acc_num = 0
    filtered_cnts = 0
    model.train(True)
    parallel_model.train(True)
    total_train_len = len(train_text_list)

    if total_train_len % batch_size == 0:
        NUM_TRAIN_ITER = int(total_train_len / batch_size)
    else:
        NUM_TRAIN_ITER = int(total_train_len / batch_size) + 1

    for i in tqdm(range(NUM_TRAIN_ITER)):
        batch_sentences = train_text_list[i * batch_size: min((i + 1) * batch_size, total_train_len)]
        labels = torch.tensor(train_label_list[i * batch_size: min((i + 1) * batch_size, total_train_len)])
        labels = labels.long().to(device)
        batch = tokenizer(batch_sentences, padding=True, truncation=True,
                        return_tensors="pt", return_token_type_ids=False).to(device)
        loss, acc_num, filtered_cnt = corrupt_train_iter(model, batch, labels, optimizer, criterion, epsilon)
        epoch_loss += loss.item() * len(batch_sentences)
        epoch_acc_num += acc_num
        filtered_cnts += filtered_cnt

    return epoch_loss / total_train_len, epoch_acc_num / total_train_len, filtered_cnts / total_train_len

def corrupt_train(train_data_path, model, tokenizer,
                batch_size, epochs, optimizer, criterion,
                device, seed, save_model=True, save_path=None, save_metric='loss', epsilon=0.1):
    logger.info('Seed: ' + str(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    best_valid_loss = float('inf')
    best_valid_acc = 0.0

    train_text_list, train_label_list = process_data(train_data_path, seed)
    autograd_hacks.add_hooks(model)

    for epoch in range(epochs):
        logger.info("Epoch: " + str(epoch))
        model.train(True)
        train_loss, train_acc, filter_ratio = corrupt_train_epoch(model, tokenizer, train_text_list, train_label_list,
                                            batch_size, optimizer, criterion, device, epsilon)

        logger.info(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}% | Filter ratio: {filter_ratio * 100:.2f}%')

    if save_model:
        # os.makedirs(save_path, exist_ok=True)
        torch.save(model, save_path)

logger = logging.getLogger(__name__)
logging.basicConfig(filename='corrupt_sgd.log', encoding='utf-8', level=logging.DEBUG)

if __name__ == '__main__':
    SEED = 1234
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    parser = argparse.ArgumentParser(description='model clean training')
    parser.add_argument('--ori_model_path', type=str, help='original model path', default="SST2_clean_model")
    parser.add_argument('--epochs', type=int, help='num of epochs', default=3)
    parser.add_argument('--save_model_path', type=str, help='path that new model saved in', default="corrupt_trained_model.pt")
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--epsilon', type=float, default=0.1, help='fraction of gradients to corrupt, value between 0 and 1.')
    parser.add_argument('--lr', default=2e-5, type=float, help='learning rate')
    args = parser.parse_args()

    model, parallel_model, tokenizer = process_model(args.ori_model_path, device)
    EPOCHS = args.epochs
    criterion = nn.CrossEntropyLoss()
    BATCH_SIZE = args.batch_size
    EPSILON = args.epsilon
    LR = args.lr
    optimizer = AdamW(model.parameters(), lr=LR)
    train_data_path = 'data/train.tsv'
    save_model = True
    save_path = args.save_model_path
    save_metric = 'acc'

    logger.info("="*10 + "Training model on clean dataset" + "="*10)
    corrupt_train(train_data_path, model, tokenizer,
                  BATCH_SIZE, EPOCHS, optimizer, criterion, device, SEED, save_model, save_path, save_metric, EPSILON)
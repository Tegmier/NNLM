
import random
def word_to_one_hot(word, word_to_index):
    one_hot = [0] * (len(word_to_index) - 1)
    index = word_to_index[word]
    if index != 0:
        one_hot[index - 1] = 1
    return one_hot

def count_max_sentence_length(data):
    max_word_count = 0
    for sentence in data:
        current_count = len(sentence)
        max_word_count = max_word_count if max_word_count > current_count else current_count
    return max_word_count

def sentence_padding(data):
    max_word_count = count_max_sentence_length(data)
    pad_element = ['<PAD>']
    for sentence in data:
        current_count = len(sentence)
        if current_count < max_word_count:
            sentence += pad_element * (max_word_count - current_count)
    return max_word_count, data

def sentence_segmentation(batch):


def data_loader(data, batch_size, win_size, ):
    bucket = random.sample(data, len(data))
    bucket = [bucket[i : i + batch_size] for i in range(0, len(bucket), batch_size)]
    random.shuffle(bucket)
    for batch in bucket:
        yield sentence_segmentation(batch, win_size, )

def word_to_one_hot(word, word_to_index):
    one_hot = [0] * len(word_to_index)
    index = word_to_index[word]
    one_hot[index] = 1
    return one_hot
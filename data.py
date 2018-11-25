import numpy as np
import os
from os import listdir
from os.path import isfile, join
from sklearn.model_selection import train_test_split
from incorrect_input import *


def noise_maker(sentence, threshold, vocab_int):
    letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    noisy_sentence = []
    i = 0
    while i < len(sentence):
        random = np.random.uniform(0, 1, 1)
        if random < threshold:
            noisy_sentence.append(sentence[i])
        else:
            new_random = np.random.uniform(0, 1, 1)
            if new_random > 0.7:
                if i == (len(sentence) - 1):
                    continue
                else:
                    noisy_sentence.append(sentence[i + 1])
                    noisy_sentence.append(sentence[i])
                    i += 1
            elif new_random < 0.3:
                random_letter = np.random.choice(letters, 1)[0]
                noisy_sentence.append(vocab_int[random_letter])
                noisy_sentence.append(sentence[i])
            else:
                pass
        i += 1
    return noisy_sentence


def split_text_for_sentences(text_tab):
    sentences = []
    for text in text_tab:
        for sentence in text.split('. '):
            sentences.append(sentence + '.')
    return sentences


def filter_sentences(sentences_int):
    max_length = 92
    min_length = 4
    good_sentences = []
    for sentence in sentences_int:
        if len(sentence) <= max_length and len(sentence) >= min_length:
            good_sentences.append(sentence)
    return good_sentences


def sentences_to_int(sentences, dictionary_int):
    int_sentences = []
    for sentence in sentences:
        int_sentence = []
        for character in sentence:
            int_sentence.append(dictionary_int[character])
        int_sentences.append(int_sentence)
    return int_sentences


def create_dictionary_int(text_tab):
    vocab_to_int = {}
    count = 0
    for text in text_tab:
        for character in text:
            if character not in vocab_to_int:
                vocab_to_int[character] = count
                count += 1
    codes = ['<PAD>', '<EOS>', '<GO>']
    for code in codes:
        vocab_to_int[code] = count
        count += 1
    return vocab_to_int


def create_dictionary_vocab(vocab_int):
    int_to_vocab = {}
    for character, value in vocab_int.items():
        int_to_vocab[value] = character
    return int_to_vocab


def load_single_file(path):
    input_file = os.path.join(path)
    with open(input_file, encoding="utf8") as input:
        file = input.read()
    return file


def load_all_files(folder_path='./txt_files'):
    path = folder_path
    files = []
    filenames = [f for f in listdir(path) if isfile(join(path, f))]
    filenames = filenames[0:]
    for single_pdf_name in filenames:
            files.append(load_single_file(path+'/'+single_pdf_name))
    return files


texts_from_files = load_all_files()
texts_from_files = clean_sentences(texts_from_files)
vocab_int = create_dictionary_int(texts_from_files)
vocab = create_dictionary_vocab(vocab_int)
sentences = split_text_for_sentences(texts_from_files)
sentences_filtered = filter_sentences(sentences_to_int(sentences, vocab_int))
training_data, testing_data = train_test_split(sentences_filtered, test_size=0.15, random_state=2)

training_data.sort(key=lambda sentence: len(sentence))
testing_data.sort(key=lambda sentence: len(sentence))

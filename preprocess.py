# preprocess.py
import numpy as np
import _pickle as pickle

num_samples = 145437

def extractChar(data_path, exchangeLanguage=False):
    input_texts, target_texts = [], []
    input_characters, target_characters = set(), set()
    lines = open(data_path, encoding='utf-8').read().split('\n')
    print(str(len(lines) - 1))

    for line in lines[: min(num_samples, len(lines) - 1)]:
        if not line.strip():
            continue
        parts = line.split('\t')
        if len(parts) != 2:
            continue
        if not exchangeLanguage:
            input_text, target_text = parts
        else:
            target_text, input_text = parts

        target_text = '\t' + target_text + '\n'
        input_texts.append(input_text)
        target_texts.append(target_text)

        input_characters.update(input_text)
        target_characters.update(target_text)

    input_characters = sorted(list(input_characters))
    target_characters = sorted(list(target_characters))
    return input_characters, target_characters, input_texts, target_texts

def encodingChar(input_characters, target_characters, input_texts, target_texts):
    num_encoder_tokens = len(input_characters)
    num_decoder_tokens = len(target_characters)
    max_encoder_seq_length = max([len(txt) for txt in input_texts])
    max_decoder_seq_length = max([len(txt) for txt in target_texts])

    print('Number of samples:', len(input_texts))
    print('Number of unique input tokens:', num_encoder_tokens)
    print('Number of unique output tokens:', num_decoder_tokens)
    print('Max sequence length for inputs:', max_encoder_seq_length)
    print('Max sequence length for outputs:', max_decoder_seq_length)

    input_token_index = dict([(char, i) for i, char in enumerate(input_characters)])
    target_token_index = dict([(char, i) for i, char in enumerate(target_characters)])

    encoder_input_data = np.zeros((len(input_texts), max_encoder_seq_length, num_encoder_tokens), dtype='float32')
    decoder_input_data = np.zeros((len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype='float32')
    decoder_target_data = np.zeros((len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype='float32')

    for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
        for t, char in enumerate(input_text):
            encoder_input_data[i, t, input_token_index[char]] = 1.
        for t, char in enumerate(target_text):
            decoder_input_data[i, t, target_token_index[char]] = 1.
            if t > 0:
                decoder_target_data[i, t - 1, target_token_index[char]] = 1.

    return encoder_input_data, decoder_input_data, decoder_target_data, input_token_index, target_token_index, num_encoder_tokens, num_decoder_tokens, max_encoder_seq_length

def prepareData(data_path):
    input_characters, target_characters, input_texts, target_texts = extractChar(data_path)
    return encodingChar(input_characters, target_characters, input_texts, target_texts)

def saveChar2encoding(filename, input_token_index, max_encoder_seq_length, num_encoder_tokens, reverse_target_char_index, num_decoder_tokens, target_token_index):
    with open(filename, "wb") as f:
        pickle.dump(input_token_index, f)
        pickle.dump(max_encoder_seq_length, f)
        pickle.dump(num_encoder_tokens, f)
        pickle.dump(reverse_target_char_index, f)
        pickle.dump(num_decoder_tokens, f)
        pickle.dump(target_token_index, f)

def getChar2encoding(filename):
    with open(filename, "rb") as f:
        input_token_index = pickle.load(f)
        max_encoder_seq_length = pickle.load(f)
        num_encoder_tokens = pickle.load(f)
        reverse_target_char_index = pickle.load(f)
        num_decoder_tokens = pickle.load(f)
        target_token_index = pickle.load(f)
    return input_token_index, max_encoder_seq_length, num_encoder_tokens, reverse_target_char_index, num_decoder_tokens, target_token_index

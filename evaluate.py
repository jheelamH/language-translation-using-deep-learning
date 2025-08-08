# evaluate.py
import numpy as np
from keras.models import load_model
from preprocess import getChar2encoding
import _pickle as pickle


def load_inference_models(encoder_path, decoder_path):
    encoder_model = load_model(encoder_path)
    decoder_model = load_model(decoder_path)
    return encoder_model, decoder_model


def decode_sequence(input_seq, encoder_model, decoder_model, target_token_index, reverse_target_char_index,
                    num_decoder_tokens, max_decoder_seq_length):
    states_value = encoder_model.predict(input_seq)

    target_seq = np.zeros((1, 1, num_decoder_tokens))
    target_seq[0, 0, target_token_index['\t']] = 1.

    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char

        if sampled_char == '\n' or len(decoded_sentence) > max_decoder_seq_length:
            stop_condition = True

        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

        states_value = [h, c]

    return decoded_sentence


def evaluate(input_seq, encoder_model, decoder_model, target_token_index, reverse_target_char_index,
             num_decoder_tokens, max_decoder_seq_length):
    decoded_sentence = decode_sequence(input_seq, encoder_model, decoder_model, target_token_index,
                                       reverse_target_char_index, num_decoder_tokens, max_decoder_seq_length)
    return decoded_sentence


if __name__ == "__main__":
    encoder_model, decoder_model = load_inference_models('encoder_model.h5', 'decoder_model.h5')

    input_token_index, max_encoder_seq_length, num_encoder_tokens, reverse_target_char_index, num_decoder_tokens, target_token_index = getChar2encoding("char_indices.pkl")

    # Example usage:
    test_input = np.zeros((1, max_encoder_seq_length, num_encoder_tokens), dtype='float32')
    test_text = "hello"
    for t, char in enumerate(test_text):
        if char in input_token_index:
            test_input[0, t, input_token_index[char]] = 1.

    translation = evaluate(test_input, encoder_model, decoder_model, target_token_index, reverse_target_char_index,
                           num_decoder_tokens, 100)
    print("Input:", test_text)
    print("Translation:", translation)

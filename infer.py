# infer.py
import numpy as np
from keras.models import Model
from keras.layers import Input, LSTM, Dense
from preprocess import prepareData, saveChar2encoding

# Paths and parameters
data_path = 'data.txt'  # Change this to your dataset path
batch_size = 64
latent_dim = 256

# Prepare the data
encoder_input_data, decoder_input_data, decoder_target_data, input_token_index, target_token_index, num_encoder_tokens, num_decoder_tokens, max_encoder_seq_length = prepareData(data_path)

# Reverse-lookup token index to decode sequences back to characters
reverse_target_char_index = dict((i, char) for char, i in target_token_index.items())

# Save encodings for later use
saveChar2encoding("char2encoding.pkl", input_token_index, max_encoder_seq_length, num_encoder_tokens, reverse_target_char_index, num_decoder_tokens, target_token_index)

# Define the model
encoder_inputs = Input(shape=(None, num_encoder_tokens))
encoder = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
encoder_states = [state_h, state_c]

decoder_inputs = Input(shape=(None, num_decoder_tokens))
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# Train the model
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=batch_size,
          epochs=30,
          validation_split=0.2)

# Save the model
model.save('seq2seq_translation.h5')

# Define sampling models for inference
encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

# Reuse the decoder LSTM
decoder_outputs, state_h, state_c = decoder_lstm(
    decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)

# Define decoder model
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)

# Save inference models
encoder_model.save('encoder_modelPredTranslation.h5')
decoder_model.save('decoder_modelPredTranslation.h5')

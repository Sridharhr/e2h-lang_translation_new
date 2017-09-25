"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
import helper
import problem_unittests as tests

"""
This cell is meant to process the data into familiar format
"""
"""
hindi_en_filename = 'data/hindencorp05.plaintext'
with open(hindi_en_filename, encoding='utf-8') as f:
    content = f.read()
content_lines = content.split('\n')
count = 0
hindi_file = open('data/hindi_small_50k','w',encoding='utf-8')
en_file = open('data/english_small_50k','w',encoding='utf-8')
for i in range(0, len(content_lines)-1):
    temp = content_lines[i].split('\t')
    align_type = temp[1]
    en = temp[3]
    hi = temp[4]
    if align_type == '1-1' and len(en) > 5 and len(hi) > 5:
        if count < 50000 - 1:
            hindi_file.write(hi + ' .\n')
            en_file.write(en + ' .\n')
            count = count + 1
        else:
            print(count)
            break
en_file.close()
hindi_file.close()
"""

#source_path = 'dl-language-translation/data/small_vocab_en'
#target_path = 'dl-language-translation/data/small_vocab_fr'
#source_path = 'data/english_small_50k'
#target_path = 'data/hindi_small_50k'
#source_path = 'data/en_all.txt'
#target_path = 'data/hi_all.txt'
source_path = 'data/hi_18sep.txt'
target_path = 'data/en_18sep.txt'
source_text = helper.load_data(source_path)
target_text = helper.load_data(target_path)

import numpy as np

def text_to_ids(source_text, target_text, source_vocab_to_int, target_vocab_to_int):
    """
    Convert source and target text to proper word ids
    :param source_text: String that contains all the source text.
    :param target_text: String that contains all the target text.
    :param source_vocab_to_int: Dictionary to go from the source words to an id
    :param target_vocab_to_int: Dictionary to go from the target words to an id
    :return: A tuple of lists (source_id_text, target_id_text)
    """
    # TODO: Implement Function
    sentences = source_text.split('\n')
    source_id_text = [[source_vocab_to_int[word] for word in line.split()] for line in sentences]
    
    sentences = target_text.split('\n')
    target_id_text = [[target_vocab_to_int[word] for word in line.split()]+[target_vocab_to_int['<EOS>']] for line in sentences]

    return source_id_text, target_id_text

PREPROCESS_PATH = '18sep_reverse_preprocess.p'
PARAM_PATH = '18sep_reverse_param.p'
save_path = 'checkpoints/18sep_reverse'

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_text_to_ids(text_to_ids)

"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
helper.preprocess_and_save_data(source_path, target_path, text_to_ids, PREPROCESS_PATH)

"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
import numpy as np
import helper
import problem_unittests as tests

(source_int_text, target_int_text), (source_vocab_to_int, target_vocab_to_int), _ = helper.load_preprocess(PREPROCESS_PATH)

"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
from distutils.version import LooseVersion
import warnings
import tensorflow as tf

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer'
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
    
def model_inputs():
    """
    Create TF Placeholders for input, targets, and learning rate.
    :return: Tuple (input, targets, learning rate, keep probability)
    """
    # TODO: Implement Function
    input_data = tf.placeholder(tf.int32, [None, None], name="input")
    targets = tf.placeholder(tf.int32, [None, None])
    lr = tf.placeholder(tf.float32)
    keep_prob = tf.placeholder(tf.float32, name="keep_prob")
    return input_data, targets, lr, keep_prob

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_model_inputs(model_inputs)

def process_decoding_input(target_data, target_vocab_to_int, batch_size):
    """
    Preprocess target data for dencoding
    :param target_data: Target Placehoder
    :param target_vocab_to_int: Dictionary to go from the target words to an id
    :param batch_size: Batch Size
    :return: Preprocessed target data
    """
    # TODO: Implement Function
    ending = tf.strided_slice(target_data, [0, 0], [batch_size, -1], [1, 1])
    dec_input = tf.concat([tf.fill([batch_size, 1], target_vocab_to_int['<GO>']), ending], 1)

    return dec_input

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_process_decoding_input(process_decoding_input)

def encoding_layer(rnn_inputs, rnn_size, num_layers, keep_prob):
    """
    Create encoding layer
    :param rnn_inputs: Inputs for the RNN
    :param rnn_size: RNN Size
    :param num_layers: Number of layers
    :param keep_prob: Dropout keep probability
    :return: RNN state
    """
    # TODO: Implement Function
    
    lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)

    drop = tf.contrib.rnn.DropoutWrapper(lstm , output_keep_prob = keep_prob)

    enc_cell = tf.contrib.rnn.MultiRNNCell([drop] * num_layers)

    _, enc_state = tf.nn.dynamic_rnn(enc_cell, rnn_inputs, dtype=tf.float32)

    return enc_state

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_encoding_layer(encoding_layer)

def decoding_layer_train(encoder_state, dec_cell, dec_embed_input, sequence_length, decoding_scope,
                         output_fn, keep_prob):
    """
    Create a decoding layer for training
    :param encoder_state: Encoder State
    :param dec_cell: Decoder RNN Cell
    :param dec_embed_input: Decoder embedded input
    :param sequence_length: Sequence Length
    :param decoding_scope: TenorFlow Variable Scope for decoding
    :param output_fn: Function to apply the output layer
    :param keep_prob: Dropout keep probability
    :return: Train Logits
    """
    # TODO: Implement Function
    train_decoder_fn = tf.contrib.seq2seq.simple_decoder_fn_train(encoder_state)
    
    drop = tf.contrib.rnn.DropoutWrapper(dec_cell, output_keep_prob = keep_prob)
    
    train_pred, _, _ = tf.contrib.seq2seq.dynamic_rnn_decoder(drop, train_decoder_fn, dec_embed_input, sequence_length, scope=decoding_scope)
    
    # Apply output function
    train_logits =  output_fn(train_pred)
    
    return train_logits


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_decoding_layer_train(decoding_layer_train)

def decoding_layer_infer(encoder_state, dec_cell, dec_embeddings, start_of_sequence_id, end_of_sequence_id,
                         maximum_length, vocab_size, decoding_scope, output_fn, keep_prob):
    """
    Create a decoding layer for inference
    :param encoder_state: Encoder state
    :param dec_cell: Decoder RNN Cell
    :param dec_embeddings: Decoder embeddings
    :param start_of_sequence_id: GO ID
    :param end_of_sequence_id: EOS Id
    :param maximum_length: Maximum length of 
    :param vocab_size: Size of vocabulary
    :param decoding_scope: TensorFlow Variable Scope for decoding
    :param output_fn: Function to apply the output layer
    :param keep_prob: Dropout keep probability
    :return: Inference Logits
    """
    # TODO: Implement Function
    infer_decoder_fn = tf.contrib.seq2seq.simple_decoder_fn_inference( \
        output_fn, encoder_state, dec_embeddings, start_of_sequence_id, end_of_sequence_id, \
        maximum_length - 1, vocab_size)
    inference_logits, _, _ = tf.contrib.seq2seq.dynamic_rnn_decoder(dec_cell, infer_decoder_fn, scope=decoding_scope)
    return inference_logits

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_decoding_layer_infer(decoding_layer_infer)

def decoding_layer(dec_embed_input, dec_embeddings, encoder_state, vocab_size, sequence_length, rnn_size,
                   num_layers, target_vocab_to_int, keep_prob):
    """
    Create decoding layer
    :param dec_embed_input: Decoder embedded input
    :param dec_embeddings: Decoder embeddings
    :param encoder_state: The encoded state
    :param vocab_size: Size of vocabulary
    :param sequence_length: Sequence Length
    :param rnn_size: RNN Size
    :param num_layers: Number of layers
    :param target_vocab_to_int: Dictionary to go from the target words to an id
    :param keep_prob: Dropout keep probability
    :return: Tuple of (Training Logits, Inference Logits)
    """
    # TODO: Implement Function

    # Decoder RNNs
    dec_cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(rnn_size)] * num_layers)

    with tf.variable_scope("decoding") as decoding_scope:
        output_fn = lambda x: tf.contrib.layers.fully_connected(x, vocab_size, None, scope=decoding_scope)
        train_logits = decoding_layer_train(encoder_state, dec_cell, dec_embed_input, sequence_length, \
                                            decoding_scope, output_fn, keep_prob)

    with tf.variable_scope("decoding", reuse=True) as decoding_scope:
        inference_logits = decoding_layer_infer(encoder_state, dec_cell, dec_embeddings, \
                                                target_vocab_to_int['<GO>'], target_vocab_to_int['<EOS>'], \
                                                sequence_length, vocab_size, decoding_scope, output_fn, keep_prob)

    return train_logits, inference_logits


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_decoding_layer(decoding_layer)

def seq2seq_model(input_data, target_data, keep_prob, batch_size, sequence_length, source_vocab_size, target_vocab_size,
                  enc_embedding_size, dec_embedding_size, rnn_size, num_layers, target_vocab_to_int):
    """
    Build the Sequence-to-Sequence part of the neural network
    :param input_data: Input placeholder
    :param target_data: Target placeholder
    :param keep_prob: Dropout keep probability placeholder
    :param batch_size: Batch Size
    :param sequence_length: Sequence Length
    :param source_vocab_size: Source vocabulary size
    :param target_vocab_size: Target vocabulary size
    :param enc_embedding_size: Decoder embedding size
    :param dec_embedding_size: Encoder embedding size
    :param rnn_size: RNN Size
    :param num_layers: Number of layers
    :param target_vocab_to_int: Dictionary to go from the target words to an id
    :return: Tuple of (Training Logits, Inference Logits)
    """
    # TODO: Implement Function
    
    # Encoder embedding
    enc_embed_input = tf.contrib.layers.embed_sequence(input_data, source_vocab_size, enc_embedding_size)
    
    enc_state = encoding_layer(enc_embed_input, rnn_size, num_layers, keep_prob)
    
    dec_input = process_decoding_input(target_data, target_vocab_to_int, batch_size)
    
    # Decoder Embedding
    dec_embeddings = tf.Variable(tf.random_uniform([target_vocab_size, dec_embedding_size]))
    dec_embed_input = tf.nn.embedding_lookup(dec_embeddings, dec_input)
    
    train_logits, inference_logits = decoding_layer(dec_embed_input, dec_embeddings, enc_state, target_vocab_size, sequence_length, rnn_size, \
                   num_layers, target_vocab_to_int, keep_prob)


    return train_logits, inference_logits


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_seq2seq_model(seq2seq_model)

# Number of Epochs
epochs = 20
# Batch Size
batch_size = 32 #default 128
# RNN Size
rnn_size = 2048
# Number of Layers
num_layers = 2
# Embedding Size
encoding_embedding_size = 2048
decoding_embedding_size = 2048
# Learning Rate
learning_rate = 0.001
# Dropout Keep Probability
keep_probability = 0.7

"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
(source_int_text, target_int_text), (source_vocab_to_int, target_vocab_to_int), _ = helper.load_preprocess(PREPROCESS_PATH)
print(len(source_vocab_to_int))
print(len(target_vocab_to_int))
max_target_sentence_length = max([len(sentence) for sentence in source_int_text])

train_graph = tf.Graph()
with train_graph.as_default():
    input_data, targets, lr, keep_prob = model_inputs()
    sequence_length = tf.placeholder_with_default(max_target_sentence_length, None, name='sequence_length')
    input_shape = tf.shape(input_data)
    
    train_logits, inference_logits = seq2seq_model(
        tf.reverse(input_data, [-1]), targets, keep_prob, batch_size, sequence_length, len(source_vocab_to_int), len(target_vocab_to_int),
        encoding_embedding_size, decoding_embedding_size, rnn_size, num_layers, target_vocab_to_int)

    tf.identity(inference_logits, 'logits')
    with tf.name_scope("optimization"):
        # Loss function
        cost = tf.contrib.seq2seq.sequence_loss(
            train_logits,
            targets,
            tf.ones([input_shape[0], sequence_length]))

        # Optimizer
        optimizer = tf.train.AdamOptimizer(lr)

        # Gradient Clipping
        gradients = optimizer.compute_gradients(cost)
        capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]
        train_op = optimizer.apply_gradients(capped_gradients)
        
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
import time

def get_accuracy(target, logits):
    """
    Calculate accuracy
    """
    #print(target.shape)
    #print(logits.shape)
    #print(target_batch.shape)
    #print(target_batch)
    max_seq = max(target.shape[1], logits.shape[1])
    #print(max_seq)
    if max_seq - target.shape[1]:
        target = np.pad(
            target,
            [(0,0),(0,max_seq - target.shape[1])],
            'constant')
    if max_seq - logits.shape[1]:
        logits = np.pad(
            logits,
            [(0,0),(0,max_seq - logits.shape[1]), (0,0)],
            'constant')

    return np.mean(np.equal(target, np.argmax(logits, 2)))

train_source = source_int_text[batch_size:]
train_target = target_int_text[batch_size:]

valid_source = helper.pad_sentence_batch(source_int_text[:batch_size])
valid_target = helper.pad_sentence_batch(target_int_text[:batch_size])

with tf.Session(graph=train_graph) as sess:
    sess.run(tf.global_variables_initializer())

    for epoch_i in range(epochs):
        for batch_i, (source_batch, target_batch) in enumerate(
                helper.batch_data(train_source, train_target, batch_size)):
            start_time = time.time()
            
            _, loss = sess.run(
                [train_op, cost],
                {input_data: source_batch,
                 targets: target_batch,
                 lr: learning_rate,
                 sequence_length: target_batch.shape[1],
                 keep_prob: keep_probability})
            
            batch_train_logits = sess.run(
                inference_logits,
                {input_data: source_batch, keep_prob: 1.0})
            batch_valid_logits = sess.run(
                inference_logits,
                {input_data: valid_source, keep_prob: 1.0})
                
            train_acc = get_accuracy(target_batch, batch_train_logits)
            valid_acc = get_accuracy(np.array(valid_target), batch_valid_logits)
            end_time = time.time()
            if (batch_i % 5 == 0):
                print('Epoch {:>3} Batch {:>4}/{} - Train Accuracy: {:>6.3f}, Validation Accuracy: {:>6.3f}, Loss: {:>6.3f} Time elapsed: {:>6.3f} s'.format(epoch_i, batch_i, len(source_int_text) // batch_size, train_acc, valid_acc, loss, (end_time - start_time)))
    # Save Model
    saver = tf.train.Saver()
    saver.save(sess, save_path)
    print('Model Trained and Saved')
    helper.save_params(save_path, PARAM_PATH)
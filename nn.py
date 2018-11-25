from collections import namedtuple
from tensorflow.python.layers.core import Dense
import time
import tensorflow as tf
from data import *

epochs = 100
batch_size = 64
num_layers = 2
rnn_size = 512
embedding_size = 128
learning_rate = 0.0005
direction = 2
threshold = 0.95
keep_probability = 0.75

def model_inputs():
    with tf.name_scope('inputs'):
        inputs = tf.placeholder(tf.int32, [None, None], name='inputs')
    with tf.name_scope('targets'):
        targets = tf.placeholder(tf.int32, [None, None], name='targets')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    inputs_length = tf.placeholder(tf.int32, (None,), name='inputs_length')
    targets_length = tf.placeholder(tf.int32, (None,), name='targets_length')
    max_target_length = tf.reduce_max(targets_length, name='max_target_len')
    return inputs, targets, keep_prob, inputs_length, targets_length, max_target_length


def process_encoding_input(targets, vocab_to_int, batch_size):
    with tf.name_scope("process_encoding"):
        ending = tf.strided_slice(targets, [0, 0], [batch_size, -1], [1, 1])
        dec_input = tf.concat([tf.fill([batch_size, 1], vocab_to_int['<GO>']), ending], 1)
    return dec_input


def encoding_layer(rnn_size, sequence_length, num_layers, rnn_inputs, keep_prob, direction):
    if direction == 1:
        with tf.name_scope("RNN_Encoder_Cell_1D"):
            for layer in range(num_layers):
                with tf.variable_scope('encoder_{}'.format(layer)):
                    lstm = tf.contrib.rnn.LSTMCell(rnn_size)
                    drop = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob=keep_prob)
                    enc_output, enc_state = tf.nn.dynamic_rnn(drop,
                                                              rnn_inputs,
                                                              sequence_length,
                                                              dtype=tf.float32)
            return enc_output, enc_state

    if direction == 2:
        with tf.name_scope("RNN_Encoder_Cell_2D"):
            for layer in range(num_layers):
                with tf.variable_scope('encoder_{}'.format(layer)):
                    cell_fw = tf.contrib.rnn.LSTMCell(rnn_size)
                    cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, input_keep_prob=keep_prob)
                    cell_bw = tf.contrib.rnn.LSTMCell(rnn_size)
                    cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, input_keep_prob=keep_prob)

                    enc_output, enc_state = tf.nn.bidirectional_dynamic_rnn(cell_fw,
                                                                            cell_bw,
                                                                            rnn_inputs,
                                                                            sequence_length,
                                                                            dtype=tf.float32)
            enc_output = tf.concat(enc_output, 2)
            return enc_output, enc_state[0]


def training_decoding_layer(dec_embed_input, targets_length, dec_cell, initial_state, output_layer, vocab_size, max_target_length):
    with tf.name_scope("Training_Decoder"):
        training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=dec_embed_input,
                                                            sequence_length=targets_length,
                                                            time_major=False)

        training_decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell,
                                                           training_helper,
                                                           initial_state,
                                                           output_layer)

        training_logits, _, _ = tf.contrib.seq2seq.dynamic_decode(training_decoder,
                                                                  output_time_major=False,
                                                                  impute_finished=True,
                                                                  maximum_iterations=max_target_length)
        return training_logits


def inference_decoding_layer(embeddings, start_token, end_token, dec_cell, initial_state, output_layer, max_target_length, batch_size):
    with tf.name_scope("Inference_Decoder"):
        start_tokens = tf.tile(tf.constant([start_token], dtype=tf.int32), [batch_size], name='start_tokens')

        inference_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embeddings,
                                                                    start_tokens,
                                                                    end_token)

        inference_decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell,
                                                            inference_helper,
                                                            initial_state,
                                                            output_layer)

        inference_logits, _, _ = tf.contrib.seq2seq.dynamic_decode(inference_decoder,
                                                                   output_time_major=False,
                                                                   impute_finished=True,
                                                                   maximum_iterations=max_target_length)
        return inference_logits


def decoding_layer(dec_embed_input, embeddings, enc_output, enc_state, vocab_size, inputs_length, targets_length,
                   max_target_length, rnn_size, vocab_to_int, keep_prob, batch_size, num_layers, direction):

    with tf.name_scope("RNN_Decoder_Cell"):
        for layer in range(num_layers):
            with tf.variable_scope('decoder_{}'.format(layer)):
                lstm = tf.contrib.rnn.LSTMCell(rnn_size)
                dec_cell = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob=keep_prob)

    output_layer = Dense(vocab_size, kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))

    attn_mech = tf.contrib.seq2seq.BahdanauAttention(rnn_size,
                                                     enc_output,
                                                     inputs_length,
                                                     normalize=False,
                                                     name='BahdanauAttention')

    with tf.name_scope("Attention_Wrapper"):
        dec_cell = tf.contrib.seq2seq.AttentionWrapper(dec_cell, attn_mech, rnn_size)

    initial_state = dec_cell.zero_state(batch_size=batch_size,dtype=tf.float32).clone(cell_state=enc_state)

    with tf.variable_scope("decode"):
        training_logits = training_decoding_layer(dec_embed_input,
                                                  targets_length,
                                                  dec_cell,
                                                  initial_state,
                                                  output_layer,
                                                  vocab_size,
                                                  max_target_length)
    with tf.variable_scope("decode", reuse=True):
        inference_logits = inference_decoding_layer(embeddings,
                                                    vocab_to_int['<GO>'],
                                                    vocab_to_int['<EOS>'],
                                                    dec_cell,
                                                    initial_state,
                                                    output_layer,
                                                    max_target_length,
                                                    batch_size)

    return training_logits, inference_logits


def seq2seq_model(inputs, targets, keep_prob, inputs_length, targets_length, max_target_length,
                  vocab_size, rnn_size, num_layers, vocab_to_int, batch_size, embedding_size, direction):

    enc_embeddings = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1, 1))
    enc_embed_input = tf.nn.embedding_lookup(enc_embeddings, inputs)
    enc_output, enc_state = encoding_layer(rnn_size, inputs_length, num_layers, enc_embed_input, keep_prob, direction)
    dec_embeddings = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1, 1))
    dec_input = process_encoding_input(targets, vocab_to_int, batch_size)
    dec_embed_input = tf.nn.embedding_lookup(dec_embeddings, dec_input)

    training_logits, inference_logits = decoding_layer(dec_embed_input,
                                                       dec_embeddings,
                                                       enc_output,
                                                       enc_state,
                                                       vocab_size,
                                                       inputs_length,
                                                       targets_length,
                                                       max_target_length,
                                                       rnn_size,
                                                       vocab_to_int,
                                                       keep_prob,
                                                       batch_size,
                                                       num_layers,
                                                       direction)
    return training_logits, inference_logits


def pad_sentence_batch(sentence_batch):
    max_sentence = max([len(sentence) for sentence in sentence_batch])
    return [sentence + [vocab_int['<PAD>']] * (max_sentence - len(sentence)) for sentence in sentence_batch]


def get_batches(sentences, batch_size, threshold):
    for batch_i in range(0, len(sentences) // batch_size):
        start_i = batch_i * batch_size
        sentences_batch = sentences[start_i:start_i + batch_size]
        sentences_batch_noisy = []
        for sentence in sentences_batch:
            sentences_batch_noisy.append(noise_maker(sentence, threshold, vocab_int))
        sentences_batch_eos = []
        for sentence in sentences_batch:
            sentence.append(vocab_int['<EOS>'])
            sentences_batch_eos.append(sentence)
        pad_sentences_batch = np.array(pad_sentence_batch(sentences_batch_eos))
        pad_sentences_noisy_batch = np.array(pad_sentence_batch(sentences_batch_noisy))
        pad_sentences_lengths = []
        for sentence in pad_sentences_batch:
            pad_sentences_lengths.append(len(sentence))
        pad_sentences_noisy_lengths = []
        for sentence in pad_sentences_noisy_batch:
            pad_sentences_noisy_lengths.append(len(sentence))

        yield pad_sentences_noisy_batch, pad_sentences_batch, pad_sentences_noisy_lengths, pad_sentences_lengths



def build_graph(keep_prob, rnn_size, num_layers, batch_size, learning_rate, embedding_size, direction):
    tf.reset_default_graph()
    inputs, targets, keep_prob, inputs_length, targets_length, max_target_length = model_inputs()
    training_logits, inference_logits = seq2seq_model(tf.reverse(inputs, [-1]),
                                                      targets,
                                                      keep_prob,
                                                      inputs_length,
                                                      targets_length,
                                                      max_target_length,
                                                      len(vocab_int) + 1,
                                                      rnn_size,
                                                      num_layers,
                                                      vocab_int,
                                                      batch_size,
                                                      embedding_size,
                                                      direction)

    training_logits = tf.identity(training_logits.rnn_output, 'logits')

    with tf.name_scope('predictions'):
        predictions = tf.identity(inference_logits.sample_id, name='predictions')
        tf.summary.histogram('predictions', predictions)
    masks = tf.sequence_mask(targets_length, max_target_length, dtype=tf.float32, name='masks')
    with tf.name_scope("cost"):
        cost = tf.contrib.seq2seq.sequence_loss(training_logits, targets, masks)
        tf.summary.scalar('cost', cost)
    with tf.name_scope("optimze"):
        optimizer = tf.train.AdamOptimizer(learning_rate)
        gradients = optimizer.compute_gradients(cost)
        capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
        train_op = optimizer.apply_gradients(capped_gradients)
    merged = tf.summary.merge_all()
    export_nodes = ['inputs', 'targets', 'keep_prob', 'cost', 'inputs_length', 'targets_length',
                    'predictions', 'merged', 'train_op', 'optimizer']
    Graph = namedtuple('Graph', export_nodes)
    local_dict = locals()
    graph = Graph(*[local_dict[each] for each in export_nodes])

    return graph


def train(model, epochs, log_string):
    with tf.Session() as sess:
        testing_loss_summary = []
        sess.run(tf.global_variables_initializer())
        iteration = 0
        display_step = 10
        stop_early = 0
        stop = 3
        per_epoch = 1
        testing_check = (len(training_data) // batch_size // per_epoch) - 1
        for epoch_i in range(1, epochs + 1):
            batch_loss = 0
            batch_time = 0
            for batch_i, (input_batch, target_batch, input_length, target_length) in enumerate(
                    get_batches(training_data, batch_size, threshold)):
                start_time = time.time()
                summary, loss, _ = sess.run([model.merged,
                                             model.cost,
                                             model.train_op],
                                            {model.inputs: input_batch,
                                             model.targets: target_batch,
                                             model.inputs_length: input_length,
                                             model.targets_length: target_length,
                                             model.keep_prob: keep_probability})
                batch_loss += loss
                end_time = time.time()
                batch_time += end_time - start_time
                iteration += 1
                if batch_i % display_step == 0 and batch_i > 0:
                    print('Epoch {:>3}/{} Batch {:>4}/{} - Loss: {:>10.4f}, time: {:>4.2f}'.format(epoch_i, epochs, batch_i, len(training_data) // batch_size, batch_loss / display_step, batch_time))
                    batch_loss = 0
                    batch_time = 0
                if batch_i % testing_check == 0 and batch_i > 0:
                    batch_loss_testing = 0
                    batch_time_testing = 0
                    for batch_i, (input_batch, target_batch, input_length, target_length) in enumerate(
                            get_batches(training_data, batch_size, threshold)):
                        start_time_testing = time.time()
                        summary, loss = sess.run([model.merged,
                                                  model.cost],
                                                 {model.inputs: input_batch,
                                                  model.targets: target_batch,
                                                  model.inputs_length: input_length,
                                                  model.targets_length: target_length,
                                                  model.keep_prob: 1})
                        batch_loss_testing += loss
                        end_time_testing = time.time()
                        batch_time_testing += end_time_testing - start_time_testing

                    n_batches_testing = batch_i + 1
                    print('Loss on batch: {:>6.3f}, time: {:>4.2f}'.format(batch_loss_testing / n_batches_testing, batch_time_testing))
                    batch_time_testing = 0
                    testing_loss_summary.append(batch_loss_testing)
                    if batch_loss_testing <= min(testing_loss_summary):
                        print('saving generation...')
                        stop_early = 0
                        checkpoint = "./{}.ckpt".format(log_string)
                        saver = tf.train.Saver()
                        saver.save(sess, checkpoint)

                    else:
                        stop_early += 1
                        if stop_early == stop:
                            break

            if stop_early == stop:
                print("Stopping Training after no improvement for long time.")
                break


def train_rnn():
    for keep_probability in [0.75]:
        for num_layers in [2]:
            for threshold in [0.95]:
                log_string = 'kp={},nl={},th={}'.format(keep_probability, num_layers, threshold)
                model = build_graph(keep_probability, rnn_size, num_layers, batch_size, learning_rate, embedding_size, direction)
                train(model, epochs, log_string)


def text_to_ints(text):
    return [vocab_int[word] for word in text]


def correct_sentence(sentence):
    text = sentence
    text = text_to_ints(text)
    checkpoint = "./kp=0.75,nl=2,th=0.95.ckpt"
    model = build_graph(keep_probability, rnn_size, num_layers, batch_size, learning_rate, embedding_size, direction)
    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, checkpoint)

        answer_logits = sess.run(model.predictions, {model.inputs: [text] * batch_size,
                                                     model.inputs_length: [len(text)] * batch_size,
                                                     model.targets_length: [len(text) + 1],
                                                     model.keep_prob: [1.0]})[0]
    pad = vocab_int["<PAD>"]
    return "".join([vocab[i] for i in answer_logits if i != pad])


def correct_text(text):
    corrected_text = []
    for sentence in text.split('. '):
        corrected_text.append(correct_sentence(sentence))
    return corrected_text

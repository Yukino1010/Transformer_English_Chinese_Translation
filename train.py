# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 17:25:44 2021

@author: s1253
"""


import os
import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
import tensorflow.keras.backend as k

from ops import EncoderLayer, DecoderLayer, preprocess, create_masks
from tensorflow.keras.layers import Dense, Embedding, Dropout, Input



output = "transformer"

output_dir = os.path.join(output, "data")
en_vocab_file = os.path.join("data", "en_vocab")
zh_vocab_file = os.path.join("data", "zh_vocab")

if not os.path.exists(output_dir):
  os.makedirs(output_dir)
  
  
config = tfds.translate.wmt.WmtConfig(
  version=tfds.core.Version('0.0.1'),
  language_pair=("zh", "en"),
  subsets={
    tfds.Split.TRAIN: ["newscommentary_v14"]
  }
)

# loading data

builder = tfds.builder("wmt_translate", config=config)
builder.download_and_prepare(download_dir=output_dir)

# using 70% of data in the dataset
examples = builder.as_dataset(split=['train[:70%]', 'train[70%:71%]','train[71%:]'], as_supervised=True)


train_examples, val_examples, _ = examples

# building english encode table

try:
  subword_encoder_en = tfds.deprecated.text.SubwordTextEncoder.load_from_file(en_vocab_file)
  print(f"載入已建立的字典： {en_vocab_file}")
except:
  print("沒有已建立的字典，從頭建立。")
  subword_encoder_en = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
      (en.numpy() for en, _ in train_examples), 
      target_vocab_size=2**15) 
  
  subword_encoder_en.save_to_file(en_vocab_file)

print(f"字典大小：{subword_encoder_en.vocab_size}")


# building chinese encode table

try:
  subword_encoder_zh = tfds.deprecated.text.SubwordTextEncoder.load_from_file(zh_vocab_file)
  print(f"載入已建立的字典： {zh_vocab_file}")
except:
  print("沒有已建立的字典，從頭建立。")
  subword_encoder_zh = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
      (zh.numpy() for _, zh in train_examples), 
      target_vocab_size=2**15, 
      max_subword_length=1) 
   
  subword_encoder_zh.save_to_file(zh_vocab_file)

print(f"字典大小：{subword_encoder_zh.vocab_size}")



def encode(en_t, zh_t):
  en_indices = [subword_encoder_en.vocab_size] + subword_encoder_en.encode(
      en_t.numpy()) + [subword_encoder_en.vocab_size + 1]
  
  zh_indices = [subword_encoder_zh.vocab_size] + subword_encoder_zh.encode(
      zh_t.numpy()) + [subword_encoder_zh.vocab_size + 1]
  
  return en_indices, zh_indices


def tf_encode(en, zh):
    return tf.py_function(encode, [en, zh], [tf.int64 , tf.int64])


MAX_LENGTH = 35 
BATCH_SIZE = 64
BUFFER_SIZE = 200

def filter_max_length(en, zh, max_length=MAX_LENGTH):
 
  return tf.logical_and(tf.size(en) <= max_length,
                        tf.size(zh) <= max_length)


def map_inp(inp, tar):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]
    
    return (inp, tar_inp), tar_real




train_dataset = (train_examples
                 .map(tf_encode)
                 .cache()
                 .filter(filter_max_length)
                 .shuffle(BUFFER_SIZE)
                 .padded_batch(batch_size=BATCH_SIZE,padded_shapes=([-1], [-1]))
                 .prefetch(tf.data.experimental.AUTOTUNE)
                 .map(map_inp)
                 )

                

val_dataset = (val_examples
               .map(tf_encode)
               .filter(filter_max_length)
               .padded_batch(BATCH_SIZE, padded_shapes=([-1], [-1]))
               .map(map_inp)
               )
 
    



#%%

# build model

num_layers = 4
d_model = 512
dff = 2048
num_heads = 8
dropout_rate = 0.2
    
input_vocab_size = subword_encoder_en.vocab_size + 2
target_vocab_size = subword_encoder_zh.vocab_size + 2
         
    
encoder_inp = Input((None,))
decoder_inp = Input((None,))
''' ---------------------------------------------------------------- '''
#encoder

enc_padding_mask, combined_mask, dec_padding_mask = create_masks()(encoder_inp, decoder_inp)

enc_len = tf.shape(encoder_inp)[1]

x = Embedding(input_vocab_size, d_model)(encoder_inp)
x = preprocess()(x, enc_len, d_model)
x = Dropout(dropout_rate)(x)

for _ in range(num_layers):
    enc_lay = EncoderLayer(d_model, num_heads, dff, dropout_rate) 
    x = enc_lay(x, mask=enc_padding_mask)

enc_out = x

''' ---------------------------------------------------------------- '''
# decoder

attention_weights = {}

dec_len = tf.shape(decoder_inp)[1]

x = Embedding(input_vocab_size, d_model)(decoder_inp)
x = preprocess()(x, dec_len, d_model)
x = Dropout(dropout_rate)(x)

for i in range(num_layers):
    dec_lay = DecoderLayer(d_model, num_heads, dff, dropout_rate) 
    x, block1, block2= dec_lay(x, enc_out, combined_mask, dec_padding_mask)
    
    attention_weights['decoder_layer{}_block1'.format(i + 1)] = block1
    attention_weights['decoder_layer{}_block2'.format(i + 1)] = block2

dec_out = x

''' ---------------------------------------------------------------- '''

Transformer_out = Dense(target_vocab_size)(x)

Transformer = Model([encoder_inp, decoder_inp], Transformer_out)

plot_model(Transformer, r"result/model_structure.png", show_shapes=True, show_layer_names=True)

#%%

# training

from tensorflow.keras.callbacks import ModelCheckpoint

# save model

checkpoint = ModelCheckpoint(r"model/origin/weights{epoch:02d}-{loss:.2f}.h5", monitor='loss', verbose=1, save_best_only=True,
mode='min')


loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')


def loss_function(real, pred):
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  
  loss_ = loss_object(real, pred)
  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask   
  
  return tf.reduce_mean(loss_)


optimizer = tf.keras.optimizers.Adam(0.0001, beta_1=0.9, beta_2=0.98, 
                                     epsilon=1e-9)


Transformer.compile(loss=loss_function, optimizer=optimizer, metrics=["accuracy"])


History = Transformer.fit(train_dataset
                        , validation_data=val_dataset
                        , epochs=50
                        , callbacks=checkpoint
                        , shuffle=True)


attention_weights_fn = k.function([encoder_inp, decoder_inp],[attention_weights])



import matplotlib.pyplot as plt

# visualize loss and accuracy

arr = History.history['accuracy'][-1]
val = History.history['val_accuracy'][-1]


plt.figure(figsize = (15,5))
plt.subplot(1,2,1)
plt.plot(History.history['accuracy'])
plt.plot(History.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')


plt.subplot(1,2,2)
plt.plot(History.history['loss'])
plt.plot(History.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')

plt.savefig(r"result/accracy_loss.png")

#%%
 
# evaluate

Transformer.load_weights(r"model/origin/weights50-0.41.h5")
Transformer.summary()

def evaluate(inp_sentence):
  
    start_token = [subword_encoder_en.vocab_size]
    end_token = [subword_encoder_en.vocab_size + 1]
    
    inp_sentence = start_token + subword_encoder_en.encode(inp_sentence) + end_token
    encoder_input = tf.expand_dims(inp_sentence, 0)
    
    decoder_input = [subword_encoder_zh.vocab_size]
    output = tf.expand_dims(decoder_input, 0)  
  
    for i in range(MAX_LENGTH):

        predictions = Transformer.predict((encoder_input, output))
        attention_weights = attention_weights_fn((encoder_input, output))[0]
        
        predictions = predictions[: , -1:, :]  # (batch_size, 1, vocab_size)
        
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
        
        if tf.equal(predicted_id, subword_encoder_zh.vocab_size + 1):
          return tf.squeeze(output, axis=0), attention_weights
        
        output = tf.concat([output, predicted_id], axis=-1)
    
    return tf.squeeze(output, axis=0), attention_weights




sentence = "In recent years, Taiwan’s economy is growing rapidly"


predicted_seq,  attention_weights_after = evaluate(sentence)

attention_weights_after = attention_weights_after[0]

target_vocab_size = subword_encoder_zh.vocab_size

predicted_seq = [idx for idx in predicted_seq if idx < target_vocab_size]

predicted_sentence = subword_encoder_zh.decode(predicted_seq)

print("sentence:", sentence)
print("-" * 20)

print("predicted_seq:", predicted_seq)
print("-" * 20)

print("predicted_sentence:", predicted_sentence)



import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']

# visualize maltihead attention

def plot_attention_weights(attention_weights, sentence, predicted_seq, layer_name, max_len_tar=None):
    
    fig = plt.figure(figsize=(17, 7))
    
    sentence = subword_encoder_en.encode(sentence)
    
   
    max_len_tar = len(predicted_seq)
    
    attention_weights = tf.squeeze(attention_weights[layer_name], axis=0)  
    # (num_heads, tar_seq_len, inp_seq_len)
    
    for head in range(attention_weights.shape[0]):
        ax = fig.add_subplot(2, 4, head + 1)
        
        attn_map = np.transpose(attention_weights[head][:max_len_tar, :])
        ax.matshow(attn_map, cmap='viridis')  # (inp_seq_len, tar_seq_len)
        
        
        ax.set_xticks(range(len(predicted_seq)))
        ax.set_xlim(-0.5, max_len_tar -1.5)
        
        ax.set_yticks(range(len(sentence) + 2))
        ax.set_xticklabels([subword_encoder_zh.decode([i]) for i in predicted_seq 
                            if i < subword_encoder_zh.vocab_size], fontsize=12)    
        
        ax.set_yticklabels(
            ['<start>'] + [subword_encoder_en.decode([i]) for i in sentence] + ['<end>'], 
            )
        
        ax.set_xlabel('Head {}'.format(head + 1))
        ax.tick_params(axis="x", labelsize=12)
        ax.tick_params(axis="y", labelsize=12)
  
    plt.tight_layout()
    plt.savefig(r"result/attention_weights.png")
    plt.show()
    plt.close(fig)
  

layer_name = f"decoder_layer{num_layers}_block2"

plot_attention_weights(attention_weights_after, sentence, predicted_seq, layer_name, max_len_tar=18)   







# Transformer_English_Chinese_Translation

# transformer

If you want to talk about which model has the biggest contribution to NLP (natural language processing) in recent years,<br>
Transformer will definitely be the first choice.

When the paper 「attention is all you need]」was published in 2017, it caused a big sensation at the time.

Because the transformer only uses the attention mechanism to construct the model, not like the traditional seq2seq model using RNN or CNN. 
Through the attention mechanism, it can convert sentences to to semantics well, and then the most important thing is that he can use GPU to perform parallel operations.

It is also because of the emergence of transfomer that many models with better performance have been produced,<br> such as
Bert, Gpt3 etc.

## Network Structure


## Hyperparameters

- BATCH_SIZE = 64
- num_layers = 4
- d_model = 512
- dff = 2048
- num_heads = 8
- dropout_rate = 0.2

## Data

This implementation uses the dataset 「wmt_translate」 from tensorflow dataset, <br>
but only use the sentence lenth < 35.

## Loss and Accuracy

## Result


## References
1.  ***Attention Is All You Need***  [[arxiv](https://arxiv.org/abs/1706.03762)]
2.  Transformer model for language understanding [https://www.tensorflow.org/text/tutorials/transformer]
3.  ***LeeMeng*** [https://leemeng.tw/neural-machine-translation-with-transformer-and-tensorflow2.html]



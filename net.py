import tensorflow as tf
from tensorflow.keras import layers

def cnn_zhang(params):
    "Implementation of https://arxiv.org/pdf/1510.03820.pdf. Return Keras Model"
    
    inp = layers.Input(shape=(params["max_len"],))
    emb = layers.Embedding(params["num_words"], params["embedding_size"], 
                        input_length=params["max_len"])(inp)

    pooled_outputs = []
    for r in params["region_sizes"]:
        for f in params["filters"]:
            x = layers.Conv1D(f, r, activation='relu', padding='same')(emb)
            x = layers.MaxPooling1D()(x)
            pooled_outputs.append(x)

    x = layers.Concatenate(axis=-1)(pooled_outputs)
    x = layers.Flatten()(x)
    x = layers.Dropout(params["dropout"])(x)
    x = layers.Dense(params["units_dense"], activation='relu')(x)
    x = layers.Dropout(params["dropout"])(x)
    output = layers.Dense(1, activation='sigmoid')(x)
    
    return tf.keras.Model(inp, output)
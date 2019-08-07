import tensorflow as tf
from tensorflow.keras.datasets import imdb
from nltk.corpus import stopwords


def load_imdb(p):
    "Function to load IMDB dataset with tf.data." 
    
    # Load and preprocess data
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=p["num_words"])
    
    if p["remove_stopwords"]:
        x_train = remove_stopwords(x_train)
        x_test = remove_stopwords(x_test)
    
    x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train,
                                                            value=0,
                                                            padding='post',
                                                            maxlen=p["max_len"])

    x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test,
                                                        value=0,
                                                        padding='post',
                                                        maxlen=p["max_len"])

    # Create tf.data.Dataset from arrays
    train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    test_data = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    train_data = train_data.shuffle(p["buffer_size"]).batch(p["batch_size"])
    train_data = train_data.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    test_data = test_data.batch(p["batch_size"])
    test_data = test_data.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return train_data, test_data

def remove_stopwords(data):
    # Load word index from imdb
    word_index = imdb.get_word_index()

    # Load english stop words    
    stop_words = set(stopwords.words('english'))

    # Create list with stop words index
    stop_words_index = []
    for w in stop_words:
        try:
            stop_words_index.append(word_index[w])
        except KeyError:
            pass

    # remove stop words
    for sentence in data:
        for word in sentence:
            if word in stop_words_index:
                sentence.remove(word)

    return data
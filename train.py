import tensorflow as tf
from tensorflow.keras import callbacks

from net import cnn_zhang
from data import load_imdb
import argparse
import json
import os

parser = argparse.ArgumentParser()
parser.add_argument('--json_file', default="params.json", help='json file path')
args = parser.parse_args()

with open(args.json_file) as f:
    params = json.load(f)
print("Params loaded: ", params)

train_data, test_data = load_imdb(params)

model = cnn_zhang(params)
model.summary()

model.compile(loss='binary_crossentropy',
              optimizer=tf.optimizers.Adam(learning_rate=params["lr"]),
              metrics=['accuracy'])

cb = []
cb.append(callbacks.TensorBoard(params["logs_dir"]))
checkp = callbacks.ModelCheckpoint(params["weights_dir"]+'/weights.h5', 
                                   save_best_only=True, save_weights_only=True)
cb.append(checkp)

if os.path.exists(params["weights_dir"]+'/weights.h5'):
    print("Loading weights")
    model.load_weights(params["weights_dir"]+'/weights.h5')

if not os.path.exists(params["weights_dir"]):
    os.mkdir(params["weights_dir"])

model.fit(train_data, epochs=params["epochs"], validation_data=test_data,
          callbacks=cb)

test_loss, test_acc = model.evaluate(test_data)

print(f'Test Loss: {test_loss:.4f}')
print(f'Test Accuracy: {test_acc:.4f}')

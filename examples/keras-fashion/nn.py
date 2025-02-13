import tensorflow as tf
import wandb

# logging code
run = wandb.init()
config = run.config
config.epochs = 100
config.lr = 0.01
config.layers = 3
config.dropout = 0.4
config.hidden_layer_1_size = 128

# load data
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

img_width = X_train.shape[1]
img_height = X_train.shape[2]
labels = ["T-shirt/top", "Trouser", "Pullover", "Dress",
          "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# 5 normalize data -> 83.5% to 89% accuracy
X_train = X_train.astype('float32') / 255.
X_test = X_test.astype('float32') / 255.

# one hot encode outputs
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)

num_classes = y_train.shape[1]

# create model
model = tf.keras.models.Sequential()
# 7 Commented next line and added convolution and maxpooling
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
# model.add(tf.keras.layers.Conv2D(16,(3,3)))
# model.add(tf.keras.layers.MaxPooling2D())
# model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
# << end of change >>
# 3 - Add additional hidden layer -> 76% to 83.5% val_acc
model.add(tf.keras.layers.Dense(config.hidden_layer_1_size, activation="relu"))
# 4 - Add dropout -> 83.5% to 72%, commented out
# model.add(tf.keras.layers.Dropout(config.dropout))
# 6 - Add dropout -> 89% to 89%
model.add(tf.keras.layers.Dropout(config.dropout))
# Add second hidden layer and Dropout -> 89% to 88.6%
model.add(tf.keras.layers.Dense(config.hidden_layer_1_size, activation="relu"))
model.add(tf.keras.layers.Dropout(config.dropout))
# 1 - Add activation function -> 16 to 19 % val_acc
model.add(tf.keras.layers.Dense(num_classes, activation="softmax"))
# 2 - Change loss function 'mse' to 'categorical_crossentropy' -> 19 to 76% val_acc
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])

# Fit the model
model.fit(X_train, y_train, epochs=config.epochs, validation_data=(X_test, y_test),
          callbacks=[wandb.keras.WandbCallback(data_type="image", labels=labels)])

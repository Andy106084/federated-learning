import collections

import numpy as np
import tensorflow as tf
import tensorflow_federated as tff

emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data()

NUM_CLIENTS = 10
BATCH_SIZE = 20

def preprocess(dataset):

  def batch_format_fn(element):
    """Flatten a batch of EMNIST data and return a (features, label) tuple."""
    return (tf.reshape(element['pixels'], [-1, 784]),
            tf.reshape(element['label'], [-1, 1]))

  return dataset.batch(BATCH_SIZE).map(batch_format_fn)

client_ids = sorted(emnist_train.client_ids)[:NUM_CLIENTS]
federated_train_data = [preprocess(emnist_train.create_tf_dataset_for_client(x))
  for x in client_ids
]

def create_keras_model():
  initializer = tf.keras.initializers.GlorotNormal(seed=0)
  return tf.keras.models.Sequential([
      tf.keras.layers.Input(shape=(784,)),
      tf.keras.layers.Dense(10, kernel_initializer=initializer),
      tf.keras.layers.Softmax(),
  ])

keras_model = create_keras_model()
tff_model = tff.learning.models.functional_model_from_keras(
    keras_model,
    loss_fn=tf.keras.losses.SparseCategoricalCrossentropy(),
    input_spec=federated_train_data[0].element_spec,
    metrics_constructor=collections.OrderedDict(
        accuracy=tf.keras.metrics.SparseCategoricalAccuracy
    ),
)

def initialize_fn():
  trainable_weights, _ =  tff_model.initial_weights
  return trainable_weights

def next_fn(server_weights, federated_dataset):
  # Broadcast the server weights to the clients.
  server_weights_at_client = broadcast(server_weights)

  # Each client computes their updated weights.
  client_weights = client_update(federated_dataset, server_weights_at_client)

  # The server averages these updates.
  mean_client_weights = mean(client_weights)

  # The server updates its model.
  server_weights = server_update(mean_client_weights)

  return server_weights

@tf.function
def client_update(model, dataset, initial_weights, client_optimizer):
  """Performs training (using the server model weights) on the client's dataset."""
  # Initialize the client model with the current server weights and the optimizer
  # state.
  client_weights = initial_weights.trainable
  optimizer_state = client_optimizer.initialize(
      tf.nest.map_structure(tf.TensorSpec.from_tensor, client_weights)
  )

  # Use the client_optimizer to update the local model.
  for batch in dataset:
    x, y = batch
    with tf.GradientTape() as tape:
      tape.watch(client_weights)
      # Compute a forward pass on the batch of data
      outputs = model.predict_on_batch(
          model_weights=(client_weights, ()), x=x, training=True
      )
      loss = model.loss(output=outputs, label=y)

    # Compute the corresponding gradient
    grads = tape.gradient(loss, client_weights)

    # Apply the gradient using a client optimizer.
    optimizer_state, client_weights = client_optimizer.next(
        optimizer_state, weights=client_weights, gradients=grads
    )

  return tff.learning.models.ModelWeights(client_weights, non_trainable=())

@tf.function
def server_update(model, mean_client_weights):
  """Updates the server model weights as the average of the client model weights."""
  del model  # Unused, just take the mean_client_weights.
  return mean_client_weights
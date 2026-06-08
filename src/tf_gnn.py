"""Custom TensorFlow GraphSAGE-style GNN (Node2Vec / BERT / hybrid variants).

This is a faithful refactor of the GNN used in ``BERT_GNN``, ``Node2Vec_GCNN``
and ``Paper_PerfectScores``. The math is intentionally preserved, including:

* the custom ``BatchNormalization`` that normalises by the running *variance*
  (not the standard deviation) -- kept as-is to reproduce the published numbers;
* the sigmoid output + ``BinaryCrossentropy`` training objective.

After training, ``extract_embeddings`` reads out the penultimate node
representations that feed the downstream classical classifiers.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Layer, Dense


# --------------------------------------------------------------------------- #
# Custom BatchNormalization (preserved exactly from the notebooks)
# --------------------------------------------------------------------------- #
class BatchNormalization(keras.layers.Layer):
    def build(self, input_shape):
        dim = input_shape[-1]
        self.gamma = self.add_weight(shape=(dim,), initializer=keras.initializers.ones, trainable=True)
        self.beta = self.add_weight(shape=(dim,), initializer=keras.initializers.zeros, trainable=True)
        self.var = self.add_weight(shape=(dim,), initializer=keras.initializers.ones, trainable=False)
        self.mean = self.add_weight(shape=(dim,), initializer=keras.initializers.zeros, trainable=False)

    def call(self, inputs, training=None):
        if training:
            mean, var = tf.nn.moments(inputs, axes=[i for i in range(inputs.shape.rank - 1)])
            normalized = (inputs - mean) / var
            self.var.assign(self.var * 0.9 + var * 0.1)
            self.mean.assign(self.mean * 0.9 + mean * 0.1)
        else:
            normalized = (inputs - self.mean) / self.var
        return normalized * self.gamma + self.beta


# --------------------------------------------------------------------------- #
# Aggregators
# --------------------------------------------------------------------------- #
class MeanAggregator(keras.layers.Layer):
    """Mean of neighbour messages, then matmul + non-linearity."""

    def __init__(self, units, dropout=0.0, bias=False, act=tf.nn.relu, concat=False, **kwargs):
        super().__init__(**kwargs)
        self.units, self.dropout, self.bias, self.act, self.concat = units, dropout, bias, act, concat

    def build(self, input_shape):
        input_dim = input_shape[0][-1]
        w_init = tf.random_normal_initializer()
        self.neigh_weights = tf.Variable(w_init(shape=(input_dim, self.units), dtype="float32"), trainable=True)
        self.self_weights = tf.Variable(w_init(shape=(input_dim, self.units), dtype="float32"), trainable=True)
        if self.bias:
            self.bias_init = tf.Variable(tf.zeros_initializer()(shape=(self.units,), dtype="float32"), trainable=True)
        super().build(input_shape)

    def call(self, inputs, training=None):
        self_vecs, neigh_vecs = inputs
        num_nodes = self_vecs.shape[0]
        node_indices, neighbour_messages = neigh_vecs
        neigh_means = tf.math.unsorted_segment_mean(neighbour_messages, node_indices, num_segments=num_nodes)
        from_neighs = tf.matmul(neigh_means, self.neigh_weights)
        from_self = tf.matmul(self_vecs, self.self_weights)
        output = tf.concat([from_self, from_neighs], axis=1) if self.concat else tf.add_n([from_self, from_neighs])
        if self.bias:
            output += self.bias_init
        return self.act(output)


class GCNAggregator(Layer):
    """Concatenate neighbour mean with self, then a single shared matmul."""

    def __init__(self, units, dropout=0.0, bias=False, act=tf.nn.relu, concat=False, **kwargs):
        super().__init__(**kwargs)
        self.units, self.dropout, self.bias, self.act, self.concat = units, dropout, bias, act, concat

    def build(self, input_shape):
        input_dim = input_shape[0][-1]
        init = tf.keras.initializers.GlorotNormal()
        self.W = tf.Variable(init(shape=(2 * input_dim, self.units), dtype="float32"), trainable=True)
        if self.bias:
            self.bias_init = tf.Variable(tf.keras.initializers.zeros()(shape=(self.units), dtype="float32"), trainable=True)
        super().build(input_shape)

    def call(self, inputs):
        self_vecs, neigh_vecs = inputs
        num_nodes = self_vecs.shape[0]
        node_indices, neighbour_messages = neigh_vecs
        neigh_means = tf.math.unsorted_segment_mean(neighbour_messages, node_indices, num_segments=num_nodes)
        means = tf.concat([neigh_means, self_vecs], axis=1)
        output = tf.matmul(means, self.W)
        if self.bias:
            output += self.bias_init
        return self.act(output)


class _PoolingAggregator(Layer):
    """Shared base for max/mean pooling aggregators over an MLP."""

    reduce_fn = staticmethod(tf.math.unsorted_segment_max)

    def __init__(self, units, dropout=0.0, bias=False, act=tf.nn.relu, concat=False, hidden_dim=512, **kwargs):
        super().__init__(**kwargs)
        self.units, self.dropout, self.bias, self.act, self.concat = units, dropout, bias, act, concat
        self.hidden_dim = hidden_dim
        self.mlp_layers = [Dense(units=hidden_dim, activation=tf.nn.relu)]

    def build(self, input_shape):
        input_dim = input_shape[0][-1]
        w_init = tf.random_normal_initializer()
        self.neigh_weights = tf.Variable(w_init(shape=(self.hidden_dim, self.units), dtype="float32"), trainable=True)
        self.self_weights = tf.Variable(w_init(shape=(input_dim, self.units), dtype="float32"), trainable=True)
        if self.bias:
            self.bias_init = tf.Variable(tf.zeros_initializer()(shape=(self.units,), dtype="float32"), trainable=True)
        super().build(input_shape)

    def call(self, inputs, training=None):
        self_vecs, neigh_vecs = inputs
        num_nodes = self_vecs.shape[0]
        node_indices, neighbour_messages = neigh_vecs
        h = neighbour_messages
        for layer in self.mlp_layers:
            h = layer(h)
        neigh_h = self.reduce_fn(h, node_indices, num_segments=num_nodes)
        from_neighs = tf.matmul(neigh_h, self.neigh_weights)
        from_self = tf.matmul(self_vecs, self.self_weights)
        output = tf.concat([from_self, from_neighs], axis=1) if self.concat else tf.add_n([from_self, from_neighs])
        if self.bias:
            output += self.bias_init
        return self.act(output)


class MaxPoolingAggregator(_PoolingAggregator):
    reduce_fn = staticmethod(tf.math.unsorted_segment_max)


class MeanPoolingAggregator(_PoolingAggregator):
    reduce_fn = staticmethod(tf.math.unsorted_segment_mean)


class TwoMaxLayerPoolingAggregator(_PoolingAggregator):
    """Two-layer MLP (512 -> 256) before max-pooling."""

    def __init__(self, units, dropout=0.0, bias=False, act=tf.nn.relu, concat=False, **kwargs):
        super().__init__(units, dropout, bias, act, concat, hidden_dim=256, **kwargs)
        self.mlp_layers = [
            Dense(units=512, activation=tf.nn.relu),
            Dense(units=256, activation=tf.nn.relu),
        ]


class SeqAggregator(Layer):
    """GRU over [self, neighbour] messages."""

    def __init__(self, units, dropout=0.0, bias=False, act=tf.nn.relu, concat=False, **kwargs):
        super().__init__(**kwargs)
        self.units, self.dropout, self.bias, self.act, self.concat = units, dropout, bias, act, concat
        self.gru = layers.GRU(units=self.units, activation="tanh",
                              recurrent_activation="sigmoid", dropout=dropout)

    def build(self, input_shape):
        input_dim = input_shape[0][-1]
        w_init = tf.random_normal_initializer()
        self.neigh_weights = tf.Variable(w_init(shape=(input_dim, self.units), dtype="float32"), trainable=True)
        self.self_weights = tf.Variable(w_init(shape=(input_dim, self.units), dtype="float32"), trainable=True)
        if self.bias:
            self.bias_init = tf.Variable(tf.zeros_initializer()(shape=(self.units,), dtype="float32"), trainable=True)
        super().build(input_shape)

    def call(self, inputs):
        self_vecs, neigh_vecs = inputs
        num_nodes = self_vecs.shape[0]
        node_indices, neighbour_messages = neigh_vecs
        neigh_h = tf.math.unsorted_segment_max(neighbour_messages, node_indices, num_segments=num_nodes)
        from_neighs = tf.matmul(neigh_h, self.neigh_weights)
        from_self = tf.matmul(self_vecs, self.self_weights)
        output = self.gru(tf.stack([from_self, from_neighs], axis=1))
        if self.bias:
            output += self.bias_init
        return self.act(output)


_AGGREGATORS = {
    "mean": MeanAggregator,
    "gcn": GCNAggregator,
    "maxpool": MaxPoolingAggregator,
    "meanpool": MeanPoolingAggregator,
    "twomaxpool": TwoMaxLayerPoolingAggregator,
    "gated": SeqAggregator,
}


# --------------------------------------------------------------------------- #
# Graph conv layer + node classifier
# --------------------------------------------------------------------------- #
class GraphConvLayer(layers.Layer):
    def __init__(self, units, dropout_rate=0.2, aggregation_type="mean", normalize=False, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.aggregation_type = aggregation_type
        self.normalize = normalize
        if aggregation_type not in _AGGREGATORS:
            raise ValueError(f"Unknown aggregator: {aggregation_type}")
        self.aggregator = _AGGREGATORS[aggregation_type](
            self.units, act=lambda x: x, dropout=dropout_rate, bias=True
        )

    def call(self, inputs):
        node_representations, edges, edge_weights = inputs
        node_indices, neighbour_indices = edges[0], edges[1]
        neighbour_representations = tf.gather(node_representations, neighbour_indices)
        node_embeddings = self.aggregator(
            (node_representations, (node_indices, neighbour_representations))
        )
        if self.normalize:
            node_embeddings = tf.nn.l2_normalize(node_embeddings, axis=-1)
        return node_embeddings


class GNNNodeClassifier(tf.keras.Model):
    def __init__(self, graph_info, num_classes, hidden_units,
                 aggregation_type="gcn", dropout_rate=0.2, normalize=True, **kwargs):
        super().__init__(**kwargs)
        node_features, edges, edge_weights = graph_info
        self.node_features = node_features
        self.edges = edges
        self.edge_weights = edge_weights if edge_weights is not None else tf.ones(shape=edges.shape[1])
        self.edge_weights = self.edge_weights / tf.math.reduce_sum(self.edge_weights)

        self.conv1 = GraphConvLayer(hidden_units[0], dropout_rate, aggregation_type, normalize, name="graph_conv1")
        self.batch_norm1 = BatchNormalization()
        self.conv2 = GraphConvLayer(hidden_units[1], dropout_rate, aggregation_type, normalize, name="graph_conv2")
        self.batch_norm2 = BatchNormalization()
        self.compute_logits = layers.Dense(units=num_classes, activation="sigmoid", name="logits")

    def call(self, input_node_indices):
        x1 = self.batch_norm1(self.conv1((self.node_features, self.edges, self.edge_weights)))
        x2 = self.batch_norm2(self.conv2((x1, self.edges, self.edge_weights)))
        input_node_indices = tf.cast(input_node_indices, dtype=tf.int32)
        node_embeddings = tf.gather(x2, input_node_indices)
        return self.compute_logits(node_embeddings)


# --------------------------------------------------------------------------- #
# Graph info + training helpers
# --------------------------------------------------------------------------- #
def build_graph_info(final_df: pd.DataFrame, feature_cols: List[str], graph_edgelist: pd.DataFrame):
    """Build the (node_features, edges, edge_weights) tuple the model expects."""
    edges = graph_edgelist[["source", "target"]].to_numpy().T
    edge_weights = tf.ones(shape=edges.shape[1], dtype=tf.float32)
    node_features = tf.cast(
        final_df.sort_values("Node_id")[feature_cols].apply(pd.to_numeric).to_numpy(),
        dtype=tf.float32,
    )
    print(f"Edges shape: {edges.shape}  Nodes shape: {node_features.shape}")
    return node_features, edges, edge_weights


def run_experiment(model, x_train, y_train, x_val, y_val,
                   learning_rate=0.001, num_epochs=40, batch_size=32, device="/CPU:0"):
    """Compile + train with EarlyStopping (matches the notebooks)."""
    with tf.device(device):
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=[tf.keras.metrics.BinaryAccuracy(threshold=0.5)],
        )
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        )
        history = model.fit(
            x=x_train, y=y_train, epochs=num_epochs, batch_size=batch_size,
            validation_data=(x_val, y_val), callbacks=[early_stopping],
        )
    return history


def extract_embeddings(model, node_indices, device="/CPU:0") -> np.ndarray:
    """Read out the GNN node representations for the given node ids."""
    with tf.device(device):
        return model.predict(node_indices)

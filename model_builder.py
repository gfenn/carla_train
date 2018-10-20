import tensorflow as tf
import tensorflow.contrib.layers as layers


def _build_hidden_layer(out, hidden, layer_norm, is_training=True):
    neurons = hidden["neurons"]
    action_out = layers.fully_connected(out, num_outputs=neurons, activation_fn=None)
    if layer_norm:
        action_out = layers.layer_norm(action_out, center=True, scale=True)
    out = tf.nn.relu(action_out)

    # Try dropout
    if "dropout" in hidden:
        dropout = hidden["dropout"]
        if 0.0 < dropout < 1.0:
            out = layers.dropout(out,
                                 keep_prob=dropout,
                                 is_training=is_training)
    return out



def _cnn_to_mlp(convs, hiddens, dueling, inpt, num_actions, scope, reuse=False, layer_norm=False, is_training=True):
    with tf.variable_scope(scope, reuse=reuse):
        out = inpt
        with tf.variable_scope("convnet"):
            # for num_outputs, kernel_size, stride, dropout, max_pool in convs:
            for conv in convs:
                out = layers.convolution2d(out,
                                           num_outputs=conv["num_outputs"],
                                           kernel_size=conv["kernel_size"],
                                           stride=conv["stride"],
                                           activation_fn=tf.nn.relu)
                if "max_pool" in conv:
                    max_pool = conv["max_pool"]
                    out = layers.max_pool2d(out,
                                            kernel_size=max_pool["size"],
                                            stride=max_pool["stride"])
        conv_out = layers.flatten(out)
        with tf.variable_scope("action_value"):
            action_out = conv_out
            for hidden in hiddens:
                action_out = _build_hidden_layer(out=action_out,
                                                 hidden=hidden,
                                                 layer_norm=layer_norm,
                                                 is_training=is_training)
            action_scores = layers.fully_connected(action_out, num_outputs=num_actions, activation_fn=None)

        if dueling:
            with tf.variable_scope("state_value"):
                state_out = conv_out
                for hidden in hiddens:
                    state_out = _build_hidden_layer(out=state_out,
                                                    hidden=hidden,
                                                    layer_norm=layer_norm,
                                                    is_training=is_training)
                state_score = layers.fully_connected(state_out, num_outputs=1, activation_fn=None)
            action_scores_mean = tf.reduce_mean(action_scores, 1)
            action_scores_centered = action_scores - tf.expand_dims(action_scores_mean, 1)
            q_out = state_score + action_scores_centered
        else:
            q_out = action_scores
        return q_out


def cnn_to_mlp(convs, hiddens, dueling=False, layer_norm=False, is_training=True):
    """This model takes as input an observation and returns values of all actions.

    Parameters
    ----------
    convs: [(int, int int)]
        list of convolutional layers in form of
        (num_outputs, kernel_size, stride)
    hiddens: [int]
        list of sizes of hidden layers
    dueling: bool
        if true double the output MLP to compute a baseline
        for action scores

    Returns
    -------
    q_func: function
        q_function for DQN algorithm.
    """

    return lambda *args, **kwargs: _cnn_to_mlp(convs, hiddens, dueling, layer_norm=layer_norm, is_training=is_training, *args, **kwargs)


Optimizing a Tensor Flow Model
==============================

Optimizing the hyperparameters of a `TensorFlow <http://tensorflow.org>`_
model is no harder than any other optimization. The only difficulty would be
the multiple levels where hyperparameters are set. For example, the learning
rate is set in the training function while the number of neurons in a given
layer is set while constructing the model.

Let say we want to optimize the hyperparameters of a convolutional neural
network over bunch of parameters including the activation function per layer,
the number of neurons in each layer and even the number of layers. First, we
need a function that builds the model. ::

    import tensorflow as tf
    from tensorflow import layers

    def cnn_model(inputs, targets, dropout_keep_prob, params):
        num_output = int(targets.get_shape()[1])
        net = inputs

        # Get the number of convolution layers from the parameter set
        for i in range(0, params["num_conv_layers"]):
            with tf.variable_scope("conv_{}".format(i)):
                # Create layer using input parameters
                net = layers.conv2d(net,
                                    filters=params["conv_{}_num_outputs".format(i)],
                                    kernel_size=params["conv_{}_kernel_size".format(i)],
                                    strides=1,
                                    padding="SAME",
                                    activation=params["conv_{}_activation_fn".format(i)])

                net = layers.conv2d(net,
                                    filters=params["conv_{}_num_outputs".format(i)],
                                    kernel_size=params["conv_{}_kernel_size".format(i)],
                                    strides=1,
                                    padding="SAME",
                                    activation=params["conv_{}_activation_fn".format(i)])

            with tf.variable_scope("mp_{}".format(i)):
                net = layers.max_pooling2d(net,
                                           pool_size=params["mp_{}_kernel_size".format(i)],
                                           strides=1,
                                           padding="VALID")

        # Dropout keep probability is set a train time.
        net = tf.nn.dropout(net, keep_prob=dropout_keep_prob)
        net = tf.contrib.layers.flatten(net)

        # Get the number of fully connectec layers from the parameter set
        for i in range(params["num_fc_layers"]):
            with tf.variable_scope("fc_{}".format(i)):
                # Create layer using input parameters
                net = tf.contrib.layers.fully_connected(net, params["fc_{}_num_outputs".format(i)],
                                      activation_fn=params["fc_{}_activation_fn".format(i)])

                net = tf.nn.dropout(net, keep_prob=dropout_keep_prob)

        with tf.variable_scope("output_layer"):
            net = tf.contrib.layers.fully_connected(net, num_output, activation_fn=tf.identity)

        return net

Then, we need a function to train the model that also has parameters to
optimize such as the learning rate, the decay rate and the dropout keep
probability. (No, it is not the ideal train function, it is just a demo.) ::

    def score_cnn(X, y, params):
        sess = tf.InteractiveSession()

        train_steps = 20
        num_classes = y.shape[1]

        X_ = tf.placeholder(tf.float32, shape=(None,) + X.shape[1:])
        y_ = tf.placeholder(tf.float32, shape=(None, num_classes))
        keep_prob_ = tf.placeholder(tf.float32)
        lr_ = tf.placeholder(tf.float32)

        logits = cnn_model(X_, y_, keep_prob_, params)

        loss_func = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_))
        optimizer_func = tf.train.AdamOptimizer(lr_).minimize(loss_func)

        predict = tf.argmax(logits, 1)
        correct_prediction = tf.equal(predict, tf.argmax(y_, 1))

        init = tf.global_variables_initializer()
        sess.run(init)

        lr_init = params["initial_learning_rate"]
        lr_decay = params["decay_learning_rate"]
        decay_steps = params["decay_steps"]

        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)

        with sess.as_default():
            for step in range(train_steps):
                lr = lr_init * lr_decay ** (step / decay_steps)
                for i in range(0, X_train.shape[0], 128):
                    feed_dict = {lr_: lr, X_: X_train[i:i+128], y_: y_train[i:i+128], 
                                 keep_prob_: params["dropout_keep_prob"]}

                _, train_loss = sess.run([optimizer_func, loss_func], feed_dict=feed_dict)
            valid_loss = 0
            for i in range(0, X_valid.shape[0], 128):
                feed_dict = {X_: X_valid[i:i+128], y_: y_valid[i:i+128], keep_prob_: 1.0}
                valid_loss += sess.run([loss_func], feed_dict=feed_dict)[0]
            valid_loss = valid_loss / (X_valid.shape[0]//128)

        return {"loss" : valid_loss}


The flexibility of the last pieces of code comes at a price; the number of
parameters to set in the search space is quite large. The next table
summarizes all the parameters that needs to be set with their type

+----------------------------+------------+----------------------------+------------+
| **Model**                  | Type       | **Training**               | Type       |
+============================+============+============================+============+
| ``num_conv_layers``        | integer    | ``initial_learning_rate``  | float      |
+----------------------------+------------+----------------------------+------------+
| ``conv_{i}_num_outputs``   | integer    | ``decay_learning_rate``    | float      |
+----------------------------+------------+----------------------------+------------+
| ``conv_{i}_kernel_size``   | integer    | ``decay_steps``            | integer    |
+----------------------------+------------+----------------------------+------------+
| ``conv_{i}_activation_fn`` | choice     | ``dropout_keep_prob``      | float      |
+----------------------------+------------+----------------------------+------------+
| ``mp_{i}_kernel_size``     | integer    |                            |            |
+----------------------------+------------+----------------------------+------------+
| ``num_fc_layers``          | integer    |                            |            |
+----------------------------+------------+----------------------------+------------+
| ``fc_{i}_num_outputs``     | integer    |                            |            |
+----------------------------+------------+----------------------------+------------+
| ``fc_{i}_activation_fn``   | choice     |                            |            |
+----------------------------+------------+----------------------------+------------+

Since there are so many hyperparameters, lets just define a function that will
creates the search space. The four training hyperparameters will sit a the top
level of our space and the two defining the number of layers will constitute
our conditions. All others will be set for these conditions. ::

    import chocolate as choco

    max_num_conv_layers = 8
    max_num_fc_layers = 3

    def create_space():
        space = {"initial_learning_rate" : choco.log(low=-5, high=-2, base=10),
                 "decay_learning_rate" : choco.uniform(low=0.7, high=1.0),
                 "decay_steps" : choco.quantized_log(low=2, high=4, step=1, base=10),
                 "dropout_keep_prob" : choco.uniform(low=0.5, high=0.95)}

        num_conv_layer_cond = dict()
        for i in range(1, max_num_conv_layers):
            condition = dict()
            for j in range(i):
                condition["conv_{}_num_outputs".format(j)] = choco.quantized_log(low=3, high=10, step=1, base=2)
                condition["conv_{}_kernel_size".format(j)] = choco.quantized_uniform(low=1, high=7, step=1)
                condition["conv_{}_activation_fn".format(j)] = choco.choice([tf.nn.relu, tf.nn.elu, tf.nn.tanh])
                condition["mp_{}_kernel_size".format(j)] = choco.quantized_uniform(low=2, high=5, step=1)

            num_conv_layer_cond[i] = condition

        space["num_conv_layers"] = num_conv_layer_cond

        num_fc_layer_cond = dict()
        for i in range(1, max_num_fc_layers):
            condition = dict()
            for j in range(i):
                condition["fc_{}_num_outputs".format(j)] = choco.quantized_log(low=3, high=10, step=1, base=2)
                condition["fc_{}_activation_fn".format(j)] = choco.choice([tf.nn.relu, tf.nn.elu, tf.nn.tanh])

            num_fc_layer_cond[i] = condition
        space["num_fc_layers"] = num_fc_layer_cond

        return space

Guess how large is the largest conditional branch of this search space. It has
34 parameters. 34 parameters is quite a lot to optimize by hand. That is why we
built Chocolate.

Ho yeah, I forgot about the last bit of code. The one that does the trick. ::

    if __name__ == "__main__":
        X, y = some_dataset()

        space = create_space()
        conn = choco.SQLiteConnection(url="sqlite:///db.db")
        sampler = choco.Bayes(conn, space, random_state=42, skip=0)

        token, params = sampler.next()
        loss = score_cnn(X, y, params)
        sampler.update(token, loss)


Nha, there was absolutly nothing new here compared to the last tutorials.

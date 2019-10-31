"""
Create and train different models, save weights and models
    Try once freezing the autoencoder layers once not

Simple Model
    Two convolutions max pool with relu one fully connected layer

Inception Module Model
    

Skip Connection Model

One with some combination of them and local response normalization

Have a function that trains and then returns a list of models and the training, testing data as well

"""

class Timer:
    """
    Utility class to time the program while running. 
    """
    def __init__(self, start_time):
        self.start_time = start_time
        self.counter = 0

    def timer(self, message=None):
        """
        Timing function that returns the time taken for this step since the starting time. Message is optional otherwise we use a counter. 
        """
        if message:
            print(f"{message} at {time.time()-self.start_time}")
        else:
            print(f"{self.counter} at {time.time()-self.start_time}")
            self.counter += 1
        return


class TensorflowModel:
    """
    A Simple CNN with a structure of Input -> Conv2D(64) -> Conv2D(32) -> MaxPooling -> Dropout -> FC(128) -> Dropout -> Output Logits(4)
    Optimizer will be Adam with a loss of crossentropy
    """

    def __init__(self):
        """
        Performs the tensorflow construction phase of the model
        """
        from datetime import datetime
        import os
        now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        self.root = "simple"
        self.logdir = "{}/run-{}/".format(self.root, now)
        self.final_model_path = os.path.join(self.root, "simple_model")

        height = 100
        width = 100
        channels = 3
        
        conv1_fmaps = 16
        conv1_ksize = 3
        conv1_stride = 1
        conv1_pad = "SAME"
        conv_dropout_rate = 0.25

        pool_fmaps = conv1_fmaps

        n_fc1 = 128
        fc1_dropout_rate = 0.5

        n_outputs = 118

        with tf.name_scope("inputs"):
            self.X = tf.placeholder(tf.float32, shape=[None, height, width, channels], name="X")
            self.y = tf.placeholder(tf.int32, shape=[None], name="y")
            self.training = tf.placeholder_with_default(False, shape=[], name='training')

        self.conv1 = tf.layers.conv2d(self.X, filters=conv1_fmaps, kernel_size=conv1_ksize,
                                 strides=conv1_stride, padding=conv1_pad,
                                 activation=tf.nn.elu, name="conv1")
        #self.conv2 = tf.layers.conv2d(self.conv1, filters=conv2_fmaps, kernel_size=conv2_ksize, strides=conv2_stride, padding=conv2_pad, activation=tf.nn.elu, name="conv2")

        with tf.name_scope("pool"):
            self.pool = tf.nn.max_pool(self.conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")
            self.pool_flat = tf.layers.flatten(self.pool, name="Flatten")
            self.pool_flat_drop = tf.layers.dropout(self.pool_flat, conv_dropout_rate, training=self.training)

        with tf.name_scope("fc1"):
            self.fc1 = tf.layers.dense(self.pool_flat_drop, n_fc1, activation=tf.nn.relu, name="fc1")
            self.fc1_drop = tf.layers.dropout(self.fc1, fc1_dropout_rate, training=self.training)

        with tf.name_scope("output"):
            self.logits = tf.layers.dense(self.fc1_drop, n_outputs, name="output")
            self.Y_proba = tf.nn.softmax(self.logits, name="Y_proba")

        with tf.name_scope("train"):
            self.xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y)
            self.loss = tf.reduce_mean(self.xentropy)
            self.loss_summary = tf.summary.scalar("Cross_Entropy", self.loss)
            self.optimizer = tf.train.AdamOptimizer()
            self.training_op = self.optimizer.minimize(self.loss)

        with tf.name_scope("eval"):
            self.correct = tf.nn.in_top_k(self.logits, self.y, 1)
            self.accuracy = tf.reduce_mean(tf.cast(self.correct, tf.float32))
            self.accuracy_summary = tf.summary.scalar("Accuracy", self.accuracy)

        with tf.name_scope("init_and_save"):
            self.init = tf.global_variables_initializer()
            self.saver = tf.train.Saver()
            self.file_writer = tf.summary.FileWriter(self.logdir, tf.get_default_graph())
        return

    def get_model_params(self):
        gvars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        return {gvar.op.name: value for gvar, value in zip(gvars, tf.get_default_session().run(gvars))}

    def restore_model_params(self, model_params):
        gvar_names = list(model_params.keys())
        assign_ops = {gvar_name: tf.get_default_graph().get_operation_by_name(gvar_name + "/Assign")
                      for gvar_name in gvar_names}
        init_values = {gvar_name: assign_op.inputs[1] for gvar_name, assign_op in assign_ops.items()}
        feed_dict = {init_values[gvar_name]: model_params[gvar_name] for gvar_name in gvar_names}
        tf.get_default_session().run(assign_ops, feed_dict=feed_dict)

    def fit(self,X_train, y_train, aug, X_valid, y_valid, n_epochs=15, batch_size=16):
        """
        Performs the tensorflow execution phase of the model
        10 seconds per batch. 100 batches per epoch = 16 minutes per epoch
        """
        steps_per_epoch = int(len(X_train)/batch_size)
        print(f"Length of X_train is {len(X_train)}, batch size is {batch_size} and so steps per epoch is {steps_per_epoch}")
        iteration = 0
        best_loss_val = np.infty
        check_interval = 500
        checks_since_last_progress = 0
        max_checks_without_progress = 20
        self.best_model_params = None 

        with tf.Session() as sess:
            self.init.run()
            epoch_timing = Timer(time.time())
            for epoch in range(n_epochs):
                iteration = 0
                steps = 0
                for X_batch, y_batch in aug.flow(X_train, y_train, batch_size=batch_size):
                    iteration += 1
                    steps += 1
                    sess.run(self.training_op, feed_dict={self.X: X_batch, self.y: y_batch, self.training: True})
                    if iteration % check_interval == 0:
                        timing.timer(f"Training Op for iteration {iteration} Done")
                        loss_val = self.loss.eval(feed_dict={self.X: X_valid, self.y: y_valid})
                        if loss_val < best_loss_val:
                            best_loss_val = loss_val
                            checks_since_last_progress = 0
                            self.best_model_params = self.get_model_params()
                        else:
                            checks_since_last_progress += 1
                    if steps >= steps_per_epoch:
                        timing.timer("Epoch")
                        break

                acc_batch = self.accuracy.eval(feed_dict={self.X: X_batch, self.y: y_batch})
                acc_val = self.accuracy.eval(feed_dict={self.X: X_valid, self.y: y_valid})
                loss_summary_str, accuracy_summary_str = sess.run([self.loss_summary, self.accuracy_summary], feed_dict={self.X:X_test, self.y:y_test})
                self.file_writer.add_summary(loss_summary_str,epoch)
                self.file_writer.add_summary(accuracy_summary_str,epoch)
                print("Epoch {}, last batch accuracy: {:.4f}%, valid. accuracy: {:.4f}%, valid. best loss: {:.6f}".format(
                          epoch, acc_batch * 100, acc_val * 100, best_loss_val))
                if checks_since_last_progress > max_checks_without_progress:
                    print("Early stopping!")
                    epoch_timing.timer("Current Epoch")

                    break
                epoch_timing.timer("Current Epoch")


            if best_model_params:
                self.restore_model_params(best_model_params)
            acc_test = self.accuracy.eval(feed_dict={self.X: X_test, self.y: y_test})
            print("Final accuracy on test set:", acc_test)
            save_path = self.saver.save(sess, self.final_model_path)
        return 
                   
    def predict(self, X):
        pass

    def evaluate(X_test, y_test):
        with tf.Session() as sess:
            self.restore_model_params(self.best_model_params)
            acc_test, loss_test = sess.run([self.accuracy, self.loss], feed_dict={self.X: X_test, self.y: y_test})
        return loss_test, acc_test
            
class KerasModel:
    def __init__(self):
        from datetime import datetime
        import os
        now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        self.fitted = False
        if self._folder_check():
            from keras.models import load_model
            self.model = load_model(self.final_model_path)
            self.fitted = True
        else:
            self.logdir = "{}/run-{}/".format(self.root, now)
            self._create_architecture()
        return

    def __str__(self):
        return self.name

    def _folder_check(self):
        raise NotImplementedError

    def _create_architechture(self):
        raise NotImplementedError

    def fit(self, X_train, y_train, aug, X_valid, y_valid, n_epochs=3, batch_size=16):
        if self.fitted:
            return
        self.model.fit_generator(aug.flow(X_train, y_train), steps_per_epoch=len(X_train)/batch_size,epochs=n_epochs, validation_data=(X_valid, y_valid))
        self.model.save(self.final_model_path)
        self.fitted = True
        return
    
    def predict(self, X):
        return self.model.predict(X)
    def evaluate(self, X_test, y_test):
        return self.model.evaluate(X_test, y_test)

class SimpleModel(KerasModel):
    def _folder_check(self):
        import os
        self.root = "simple"
        self.name = "simple_model"
        self.final_model_path = os.path.join(self.root, self.name)
        return self.name in os.listdir(self.root)

    def _create_architecture(self):
        from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization
        from keras.models import Sequential
        
        height = 100
        width = 100
        channels = 3
        # Has to be changed for flowers
        
        conv1_fmaps = 16
        conv1_ksize = 3
        conv1_stride = 1
        conv1_pad = "SAME"
        conv_dropout_rate = 0.25
        pool_fmaps = conv1_fmaps
        n_fc1 = 128
        fc1_dropout_rate = 0.5
        n_outputs = 118

        self.model = Sequential() 

        conv1 = Conv2D(conv1_fmaps, conv1_ksize, padding="SAME", input_shape=(height, width, channels), activation='elu')
        self.model.add(conv1)

        pool = MaxPooling2D()
        self.model.add(pool)

        pool_flat = Flatten()
        self.model.add(pool_flat)

        pool_dropped = Dropout(0.5)
        self.model.add(pool_dropped)


        fc = Dense(n_fc1, activation='elu')
        self.model.add(fc)

        fc_dropped = Dropout(0.5)
        self.model.add(fc_dropped)

        final = Dense(n_outputs, activation="softmax")
        self.model.add(final)

        self.model.compile("adam", loss="categorical_crossentropy", metrics=['accuracy'])
        return

class ResidualModel(KerasModel):
    def _create_architecture(self):
        """
        We simple add a skip connection to a second fully connected layer. i.e give it to a 30,000 node layer and then add the input layer flattened to this layer
        The flatten layer has a shape (None, 40000)
        """
        from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization, add, Input
        from keras.models import Model
        
        height = 100
        width = 100
        channels = 3
        # Has to be changed for flowers
        
        conv1_fmaps = 12
        conv1_ksize = 3
        conv1_stride = 1
        conv1_pad = "SAME"
        conv_dropout_rate = 0.25
        pool_fmaps = conv1_fmaps
        n_fc1 = 128
        fc1_dropout_rate = 0.5
        n_outputs = 118

        inp = Input(shape=(height, width, channels))

        conv1 = Conv2D(conv1_fmaps, conv1_ksize, padding="SAME", input_shape=(height, width, channels), activation='elu')(inp)

        pool = MaxPooling2D()(conv1)

        pool_flat = Flatten()(pool)

        input_flat = Flatten()(inp)

        final_flat = add([pool_flat, input_flat])

        dropped = Dropout(0.5) (final_flat)

        fc = Dense(n_fc1, activation='elu')(dropped)
       
        fc_dropped = Dropout(0.5)(fc)

        final = Dense(n_outputs, activation="softmax")(fc_dropped)
        
        self.model = Model(inputs=inp, outputs=final)

        self.model.compile("adam", loss="categorical_crossentropy", metrics=['accuracy'])
        return


    def _folder_check(self):
        import os
        self.root = "residual"
        self.name = "residual_model"
        self.final_model_path = os.path.join(self.root, self.name)
        return self.name in os.listdir(self.root)

class InceptionModel(KerasModel):
    def _create_architecture(self):
        """
        We add an inception module before the first convolutionary layer
        """
        from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization, add, Input
        from keras.models import Model
        
        height = 100
        width = 100
        channels = 3
        # Has to be changed for flowers
        
        conv1_fmaps = 16
        conv1_ksize = 3
        conv1_stride = 1
        conv1_pad = "SAME"
        conv_dropout_rate = 0.25
        pool_fmaps = conv1_fmaps
        n_fc1 = 128
        fc1_dropout_rate = 0.5
        n_outputs = 118

        inp = Input(shape=(height, width, channels))

        inception = self._inception_module(inp)

        conv1 = Conv2D(conv1_fmaps, conv1_ksize, padding="SAME", input_shape=(height, width, channels), activation='elu')(inception)

        pool = MaxPooling2D()(conv1)

        pool_flat = Flatten()(pool)

        dropped = Dropout(0.5) (pool_flat)

        fc = Dense(n_fc1, activation='elu')(dropped)
       
        fc_dropped = Dropout(0.5)(fc)

        final = Dense(n_outputs, activation="softmax")(fc_dropped)
        
        self.model = Model(inputs=inp, outputs=final)

        self.model.compile("adam", loss="categorical_crossentropy", metrics=['accuracy'])
        return

    def _inception_module(self, inp):
        from keras.layers import Conv2D, MaxPooling2D, Input, concatenate

        tower_1 = Conv2D(16, (1, 1), padding='same', activation='relu')(inp)
        tower_1 = Conv2D(16, (3, 3), padding='same', activation='relu')(tower_1)

        tower_2 = Conv2D(16, (1, 1), padding='same', activation='relu')(inp)
        tower_2 = Conv2D(16, (5, 5), padding='same', activation='relu')(tower_2)

        tower_3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(inp)
        tower_3 = Conv2D(16, (1, 1), padding='same', activation='relu')(tower_3)

        output = concatenate([tower_1, tower_2, tower_3], axis=1)

        return output

    def _folder_check(self):
        import os
        self.root = "inception"
        self.name = "inception"
        self.final_model_path = os.path.join(self.root, self.name)
        return self.name in os.listdir(self.root)

class HybridModel(KerasModel):
    def _folder_check(self):
        import os
        self.root = "hybrid"
        self.name = "hybrid_model"
        self.final_model_path = os.path.join(self.root, self.name)
        return self.name in os.listdir(self.root)

    def _create_architecture(self):
        """
        Heavy Option is Input to inception to conv1 to conv2 to FC with a skip from conv1 to FC too and then output
        """
        from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization
        from keras.models import Sequential
        
        height = 100
        width = 100
        channels = 3
        # Has to be changed for flowers
        
        conv1_fmaps = 16
        conv1_ksize = 3
        conv1_stride = 1
        conv1_pad = "SAME"
        conv_dropout_rate = 0.25
        pool_fmaps = conv1_fmaps
        n_fc1 = 128
        fc1_dropout_rate = 0.5
        n_outputs = 118

        self.model = Sequential() 

        conv1 = Conv2D(conv1_fmaps, conv1_ksize, padding="SAME", input_shape=(height, width, channels), activation='elu')
        self.model.add(conv1)

        pool = MaxPooling2D()
        self.model.add(pool)

        pool_flat = Flatten()
        self.model.add(pool_flat)

        pool_dropped = Dropout(0.5)
        self.model.add(pool_dropped)


        fc = Dense(n_fc1, activation='elu')
        self.model.add(fc)

        fc_dropped = Dropout(0.5)
        self.model.add(fc_dropped)

        final = Dense(n_outputs, activation="softmax")
        self.model.add(final)

        self.model.compile("adam", loss="categorical_crossentropy", metrics=['accuracy'])
        return


def final_model_generator(model_name="simple"):
    """
    Options are simple, residual and inception
    """
    import numpy as np
    import time
    from Processor import labelled_final
    X_train, X_valid, X_test, y_train, y_valid, y_test, aug = labelled_final()
    from keras.utils import to_categorical
    y_train = to_categorical(y_train)
    y_valid = to_categorical(y_valid)
    y_test = to_categorical(y_test)
    model_dict = {"simple": SimpleModel, "residual": ResidualModel, "inception":InceptionModel}
    model = model_dict[model_name]()
    #timing.timer("Fitting Started")
    model.fit(X_train, y_train, aug, X_valid, y_valid)
    print(model.evaluate(X_test, y_test))
    return model

def final_data_generator():
    from Processor import labelled_final
    X_train, X_valid, X_test, y_train, y_valid, y_test, aug = labelled_final()
    from keras.utils import to_categorical
    y_train = to_categorical(y_train)
    y_valid = to_categorical(y_valid)
    y_test = to_categorical(y_test)
    return X_train, X_valid, X_test, y_train, y_valid, y_test




if __name__ == "__main__":
    import tensorflow as tf
    import numpy as np
    import time
    #timing = Timer(time.time())
    
    from Processor import labelled_final
    X_train, X_valid, X_test, y_train, y_valid, y_test, aug = labelled_final()
    from keras.utils import to_categorical
    y_train = to_categorical(y_train)
    y_valid = to_categorical(y_valid)
    y_test = to_categorical(y_test)
    model = SimpleModel()
    #timing.timer("Fitting Started")
    model.fit(X_train, y_train, aug, X_valid, y_valid)
    print(model.evaluate(X_test, y_test))

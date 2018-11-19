import random
from sklearn.cross_validation import train_test_split
import keras
from data_augmentation import augmentation
from keras import backend
from keras.legacy import interfaces
from keras.preprocessing.image import ImageDataGenerator
from multiprocessing import Pool
from time import time
import tensorflow
import numpy as np

DEBUG = False
INIT_MAX = {'features':[2,3,4], 'classification':[1]}

class TimedStopping(keras.callbacks.Callback):
    """
    Keras Callback, used to stop the train of a network
    based on time.

    From: https://github.com/keras-team/keras/issues/1625
    Credits to kylemcdonald (https://github.com/kylemcdonald)

    Attributes
    ----------
    seconds : int
        maximum time for the train of a network
    verbose : boolean
        False does not print anything
        True prints when the train is stopped
    """

    def __init__(self, seconds=None, verbose=0):
        super(keras.callbacks.Callback, self).__init__()

        self.start_time = 0
        self.seconds = seconds
        self.verbose = verbose

    def on_train_begin(self, logs={}):
        self.start_time = time()

    def on_epoch_end(self, epoch, logs={}):
        if time() - self.start_time > self.seconds:
            self.model.stop_training = True
            if self.verbose:
                print('Stopping after %s seconds.' % self.seconds)

class CNNEvaluator:
    """
    Class used for storing the dataset, mapping the individuals into keras
    interpretable models, and evaluating the performance of the models.


    Methods
    -------
    load_cifar(n_classes=10, val_size=7500)
        loads the cifar-10 dataset
            .n_classes [in] (int): number of classes of the problem (default 10)
            .test_size [in] (int): size of the validation partition, used during
                                   evolution for assessing the performance of the
                                   networks (detaul 7500)
            .dataset [out] (dict): loaded dataset, with three partitions: train,
                                   validation and test; the test is kept out of
                                   evolution

    get_layers(phenotype)
        auxiliary function, used to partition the layers phenotype, to ease the
        mapping into a model interpretable by keras
            .phenotype [in] (str): phenotype corresponding to the layers
            .layers [out] (list): list of layers of the individual

    get_learning(learning)
        auxiliary function, used to partition the learning phenotype, to ease
        the mapping into an optimiser interpretable by keras
            .learning [in] (str): phenotype corresponding to the learning
            .learning_params [out] (dict): learning parameters and corresponding
                                           parameterisation

    assemble_network(keras_layers, input_size)
        function that maps the layers processed by @get_layers into a model
        interpretable by keras
            .keras_layers [in] (list): list of layers (corresponds to the layers
                                       from @get_layers)
            .input_size [in] (tuple): the input dimension of the network
            .model [out] (keras.models.Model): keras model, ready for train

    assemble_optimiser(learning)
        function that maps the learning parameters processed by @get_learning
        into an optimiser interpretable by keras
            .learning [in] (dict): learning parameters corresponds to the
                                   learning_params from @get_learning)
            .optimiser [out] (keras.optimizers.Optimizer): keras optimiser

    evaluate(phenotype, train_time, input_size=(32, 32, 3))
        function used to evaluate the candidate solution, based on a maximum train
        time; it also uses early stop
            .phenotype [in] (str): phenotype of the candidate solution generated 
                                   by DENSER
            .train_time [in] (int): maximum train time (in seconds)
            .input_size [in] (tuple): the shape of the input signal of the network
                                      (default to (32,32,3) as in the original paper
                                        we deal with RGB images)
            .history [out] (keras.callbacks.History): stores all the information about
                                                      the train: accuracy, loss (both
                                                      train and validation)
    """


    def __init__(self):
        self.dataset = self.load_cifar()

    def load_cifar(self, n_classes=10, val_size=7500):
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,
                                                          test_size=val_size,
                                                          stratify=y_train)

        x_train = x_train.astype('float32')
        x_val = x_val.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_val /= 255
        x_test /= 255

        x_mean = 0
        for x in x_train:
            x_mean += x
        x_mean /= len(x_train)
        x_train -= x_mean
        x_val -= x_mean
        x_test -= x_mean

        y_train = keras.utils.to_categorical(y_train, n_classes)
        y_val = keras.utils.to_categorical(y_val, n_classes)

        dataset = {'x_train': x_train, 'y_train': y_train,
                   'x_val': x_val, 'y_val': y_val,
                   'x_test': x_test, 'y_test': y_test}

        return dataset

    def get_layers(self, phenotype):
        raw_phenotype = phenotype.split(' ')

        idx = 0
        first = True
        node_type, node_val = raw_phenotype[idx].split(':')
        layers = []

        while idx < len(raw_phenotype):
            if node_type == 'layer':
                if not first:
                    layers.append((layer_type, node_properties))
                else:
                    first = False
                layer_type = node_val
                node_properties = {}
            else:
                node_properties[node_type] = node_val.split(',')

            idx += 1
            if idx < len(raw_phenotype):
                node_type, node_val = raw_phenotype[idx].split(':')

        layers.append((layer_type, node_properties))

        return layers


    def get_learning(self, learning):
        raw_learning = learning.split(' ')

        idx = 0
        learning_params = {}
        while idx < len(raw_learning):
            param_name, param_value = raw_learning[idx].split(':')
            learning_params[param_name] = param_value.split(',')
            idx += 1

        for _key_ in sorted(list(learning_params.keys())):
            if len(learning_params[_key_]) == 1:
                try:
                    learning_params[_key_] = eval(learning_params[_key_][0])
                except NameError:
                    learning_params[_key_] = learning_params[_key_][0]

        return learning_params


    def assemble_network(self, keras_layers, input_size):
        first_fc = True
        inputs = keras.layers.Input(shape=input_size)

        layers = []
        for layer_type, layer_params in keras_layers:
            if layer_type == 'conv':
                conv_layer = keras.layers.Conv2D(filters=int(layer_params['num-filters'][0]),
                                                 kernel_size=(int(layer_params['filter-shape'][0]), int(layer_params['filter-shape'][0])),
                                                 strides=(int(layer_params['stride'][0]), int(layer_params['stride'][0])),
                                                 padding=layer_params['padding'][0],
                                                 activation=layer_params['act'][0],
                                                 use_bias=eval(layer_params['bias'][0]),
                                                 kernel_initializer='he_normal',
                                                 kernel_regularizer=keras.regularizers.l2(0.0005))
                layers.append(conv_layer)

            elif layer_type == 'batch-norm':
                batch_norm = keras.layers.BatchNormalization()
                layers.append(batch_norm)

            elif layer_type == 'pool-avg':
                pool_avg = keras.layers.AveragePooling2D(pool_size=(int(layer_params['kernel-size'][0]), int(layer_params['kernel-size'][0])),
                                                                 strides=int(layer_params['stride'][0]),
                                                                 padding=layer_params['padding'][0])
                layers.append(pool_avg)

            elif layer_type == 'pool-max':
                pool_max = keras.layers.MaxPooling2D(pool_size=(int(layer_params['kernel-size'][0]), int(layer_params['kernel-size'][0])),
                                                             strides=int(layer_params['stride'][0]),
                                                             padding=layer_params['padding'][0])
                layers.append(pool_max)

            elif layer_type == 'fc':
                fc = keras.layers.Dense(int(layer_params['num-units'][0]),
                                             activation=layer_params['act'][0],
                                             use_bias=eval(layer_params['bias'][0]),
                                             kernel_initializer='he_normal',
                                             kernel_regularizer=keras.regularizers.l2(0.0005))
                layers.append(fc)

            elif layer_type == 'dropout':
                dropout = keras.layers.Dropout(rate=float(layer_params['rate'][0]))
                layers.append(dropout)



        for layer in keras_layers:
            layer[1]['input'] = map(int, layer[1]['input'])

        data_layers = []
        invalid_layers = []

        for layer_idx, layer in enumerate(layers):
            
            try:
                if len(keras_layers[layer_idx][1]['input']) == 1:
                    if keras_layers[layer_idx][1]['input'][0] == -1:
                        data_layers.append(layer(inputs))
                    else:
                        if keras_layers[layer_idx][0] == 'fc' and first_fc:
                            first_fc = False
                            flatten = keras.layers.Flatten()(data_layers[keras_layers[layer_idx][1]['input'][0]])
                            data_layers.append(layer(flatten))
                            continue

                        data_layers.append(layer(data_layers[keras_layers[layer_idx][1]['input'][0]]))

                else:
                    #Get minimum shape
                    minimum_shape = input_size[0]
                    for input_idx in keras_layers[layer_idx][1]['input']:
                        if input_idx != -1 and input_idx not in invalid_layers:
                            if data_layers[input_idx].shape[-3:][0] < minimum_shape:
                                minimum_shape = int(data_layers[input_idx].shape[-3:][0])

                    #Reshape signals to the same shape
                    merge_signals = []
                    for input_idx in keras_layers[layer_idx][1]['input']:
                        if input_idx == -1:
                            if inputs.shape[-3:][0] > minimum_shape:
                                actual_shape = int(inputs.shape[-3:][0])
                                merge_signals.append(keras.layers.MaxPooling2D(pool_size=(actual_shape-(minimum_shape-1), actual_shape-(minimum_shape-1)), strides=1)(inputs))
                            else:
                                merge_signals.append(inputs)

                        elif input_idx not in invalid_layers:
                            if data_layers[input_idx].shape[-3:][0] > minimum_shape:
                                actual_shape = int(data_layers[input_idx].shape[-3:][0])
                                merge_signals.append(keras.layers.MaxPooling2D(pool_size=(actual_shape-(minimum_shape-1), actual_shape-(minimum_shape-1)), strides=1)(data_layers[input_idx]))
                            else:
                                merge_signals.append(data_layers[input_idx])

                    if len(merge_signals) == 1:
                        merged_signal = merge_signals[0]
                    elif len(merge_signals) > 1:
                        merged_signal = keras.layers.concatenate(merge_signals)
                    else:
                        merged_signal = data_layers[-1]

                    data_layers.append(layer(merged_signal))
            except ValueError as e:
                data_layers.append(data_layers[-1])
                invalid_layers.append(layer_idx)
                if DEBUG:
                    print keras_layers[layer_idx][0]
                    print e




        model = keras.models.Model(inputs=inputs, outputs=data_layers[-1])
        
        if DEBUG:
            model.summary()

        return model


    def assemble_optimiser(self, learning):
        if learning['learning'] == 'rmsprop':
            return keras.optimizers.RMSprop(lr = float(learning['lr']),
                                            rho = float(learning['rho']),
                                            decay = float(learning['decay']))
        
        elif learning['learning'] == 'gradient-descent':
            return keras.optimizers.SGD(lr = float(learning['lr']),
                                        momentum = float(learning['momentum']),
                                        decay = float(learning['decay']),
                                        nesterov = bool(learning['nesterov']))

        elif learning['learning'] == 'adam':
            return keras.optimizers.Adam(lr = float(learning['lr']),
                                         beta_1 = float(learning['beta1']),
                                         beta_2 = float(learning['beta2']),
                                         decay = float(learning['decay'])) #,
                                         #amsgrad = bool(learning['amsgrad']))


    def evaluate(self, phenotype, train_time, input_size=(32, 32, 3)):
        model_phenotype, learning_phenotype = phenotype.split('learning:')
        learning_phenotype = 'learning:'+learning_phenotype.rstrip().lstrip()
        model_phenotype = model_phenotype.rstrip().lstrip().replace('  ', ' ')

        keras_layers = self.get_layers(model_phenotype)
        keras_learning = self.get_learning(learning_phenotype)
        batch_size = int(keras_learning['batch_size'])
        
        model = self.assemble_network(keras_layers, input_size)

        opt = self.assemble_optimiser(keras_learning)
        model.compile(loss='categorical_crossentropy',
                      optimizer=opt,
                      metrics=['accuracy'])

        datagen_train = ImageDataGenerator(preprocessing_function=augmentation) 

        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=int(keras_learning['early_stop']))
        time_stop = TimedStopping(seconds=train_time, verbose=DEBUG)


        trainable_count = int(np.sum([backend.count_params(p) for p in set(model.trainable_weights)]))

        score = model.fit_generator(datagen_train.flow(self.dataset['x_train'],
                                                       self.dataset['y_train'],
                                                       batch_size=batch_size),
                                    steps_per_epoch=(self.dataset['x_train'].shape[0]//batch_size),
                                    epochs=int(keras_learning['epochs']),
                                    validation_data=(self.dataset['x_val'], self.dataset['y_val']),
                                    callbacks = [early_stop, time_stop],
                                    verbose=0)

        if DEBUG:
            print phenotype, max(score.history['val_acc'])
        score.history['trainable_parameters'] = trainable_count

        return score.history



def evaluate(args):
    """
        Method used for performing the train of the networks in a separate process.
    """
    cnn_eval, phenotype, train_time = args

    try:
        return cnn_eval.evaluate(phenotype, train_time)
    except tensorflow.errors.ResourceExhaustedError as e:
        return None


class Module:
    """
    Class used for encoding a module (part of the individual)

    Attributes
    ----------
    module : str
        module to be encoded
    max_expansions : int
        maximum number of evolutionary units of the module
    levels_back : int
        upper bound on the layers a given layer can receive as input
    layers : list
        list of layers of the module
    connections : dict
        connections each layer receives as input

    Methods
    -------
    initialise(grammar, init_max, reuse=0.2)
        initialise the genoytpe of a candidate solution (at random)
            .grammar [in] (Grammar): grammar
            .init_max [in] [dict]: maximum number of layers when initialising
            .reuse [in] (float): probability of reusing an existing layer, instead
                                 of randomly generating a new one (default 0.2)    
    """

    def __init__(self, module, max_expansions, levels_back):
        self.module = module
        self.max_expansions = max_expansions
        self.levels_back = levels_back
        self.layers = []
        self.connections = {}

    def initialise(self, grammar, init_max, reuse=0.2):
        num_expansions = random.choice(init_max[self.module])

        #Initialise layers
        for idx in range(num_expansions):
            if idx>0 and random.random() <= reuse:
                r_idx = random.randint(0, idx-1)
                self.layers.append(self.layers[r_idx])
            else:
                self.layers.append(grammar.initialise(self.module))

        #Initialise connections: feed-forward and allowing skip-connections
        self.connections = {}
        for layer_idx in range(num_expansions):
            if layer_idx == 0:
                self.connections[layer_idx] = [-1,]
            else:
                connection_possibilities = list(range(max(0, layer_idx-self.levels_back), layer_idx-1))
                if len(connection_possibilities) < self.levels_back-1:
                    connection_possibilities.append(-1)

                sample_size = random.randint(0, len(connection_possibilities))
                
                self.connections[layer_idx] = [layer_idx-1] 
                if sample_size > 0:
                    self.connections[layer_idx] += random.sample(connection_possibilities, sample_size)


class Individual:
    """
    Class used for encoding an individual: each individual is composed of
    multiple modules, a macro structure and an output layer.

    Attributes
    ----------
    network_structure : list
        structure of the network
    output_rule : str
        defines the last layer of the network (kept static because of 
        the required fixed number of units of the last layer)
    macro_rules: list
        macro structure of the network
    modules: list
        list of modules of the network, each module corresponds
        to a position of the network structure
    output: grammatical derivation
        genotype of the output_rule
    macro: list of grammatical derivations
        genotype of the macro_rules
    phenotype: str
        phenotype of the individual
    fitness: float
        fitness value, in the current scenario, the maximum validation
        accuracy on the train
    metrics: dict
        several evaluation metrics: keeps the history
    num_epochs: int
        number of train epochs of the last train
    trainable_parameters: int
        number of trainable paramters of the network
    time: int
        time the last train took (seconds)

    Methods
    -------
    initialise(grammar, levels_back, init_max, reuse=0.2)
        initialise the genoytpe of a candidate solution (at random)
            .grammar [in] (Grammar): grammar
            .levels_back [in] (dict): maximum number of levels back, i.e.,
                                      upper bound on the layers a given layer
                                      can receive as input
            .init_max [in] [dict]: maximum number of layers when initialising
            .reuse [in] (float): probability of reusing an existing layer, instead
                                 of randomly generating a new one (default 0.2)
            .self [out] (Individual): return the individual 
    
    decode(grammar)
        performs the genotype to phenotype mapping
            .grammar [in] (Grammar): grammar
            .phenotype [out] (str): phenotype

    evaluate(grammar, cnn_eval, train_time):
        evalautes the individual
            .grammar [in] (Grammar): grammar
            .cnn_eval [in] (CNNEvaluator): evaluator for the networks
            .train_time [in] (int): maximum train time (in seconds)
            .fitness [out] [float]: fitness
    """

    def __init__(self, network_structure, macro_rules, output_rule):
        self.network_structure = network_structure
        self.output_rule = output_rule
        self.macro_rules = macro_rules
        self.modules = []
        self.output = None
        self.macro = []
        self.phenotype = None
        self.fitness = None
        self.metrics = None
        self.num_epochs = None
        self.trainable_parameters = None
        self.time = None

    def initialise(self, grammar, levels_back, init_max, reuse=0.2):
        for non_terminal, max_expansions in self.network_structure:
            new_module = Module(non_terminal, max_expansions, levels_back[non_terminal])
            new_module.initialise(grammar, init_max)

            self.modules.append(new_module)

        #Initialise output
        self.output = grammar.initialise(self.output_rule)

        # Initialise the macro structure: learning, data augmentation, etc.
        for rule in self.macro_rules:
            self.macro.append(grammar.initialise(rule))

        return self

    def decode(self, grammar):
        phenotype = ''
        offset = 0
        layer_counter = 0
        for module in self.modules:
            offset = layer_counter
            for layer_idx, layer_genotype in enumerate(module.layers):
                layer_counter += 1
                phenotype += ' ' + grammar.decode(module.module, layer_genotype)+ ' input:'+",".join(map(str, np.array(module.connections[layer_idx])+offset))

        phenotype += ' '+grammar.decode(self.output_rule, self.output)+' input:'+str(layer_counter-1)

        for rule_idx, macro_rule in enumerate(self.macro_rules):
            phenotype += ' '+grammar.decode(macro_rule, self.macro[rule_idx])

        self.phenotype = phenotype.rstrip().lstrip()
        return self.phenotype

    def evaluate(self, grammar, cnn_eval, train_time):
        phenotype = self.decode(grammar)
        start = time()
        pool = Pool(processes=1)

        result = pool.apply_async(evaluate, [(cnn_eval, phenotype, train_time)])
        pool.close()
        pool.join()
        metrics = result.get()

        if metrics is not None:
            self.metrics = metrics
            self.fitness = max(self.metrics['val_acc'])
            self.num_epochs = len(self.metrics['val_acc'])
            self.trainable_parameters = self.metrics['trainable_parameters']
        else:
            self.metrics = None
            self.fitness = -1
            self.num_epochs = -1
            self.trainable_parameters = -1

        self.time = time() - start

        return self.fitness

from sys import argv
import random
import numpy as np
from grammar import Grammar
from utils import CNNEvaluator, Individual
from copy import deepcopy
from json import dumps
from os import makedirs
import pickle

class Config:
    """
    Class used for storing the configuration parameters.
    TODO: will be changed to a config file.

    Attributes
    ----------
    random_seeds : list
        seeds used by random
    numpy_seeds : list
        seeds used by numpy
    num_generations : int
        maximum number of generations of the (1+lambda)-ES
    _lambda : int
        number of individuals to generate in each generation
    
    add_layer : float
        probability of adding a new layer
    reuse_layer : float
        probability of reusing an existing layer, instead of randomly 
        generating a new one 
    remove_layer : float
        probability of removing a layer
    add_connection : float
        probability of adding a connection
    remove_connection : float
        probability of removing a connection
    dsge_layer : float
        probability of chaning a parameter of one of the layers
    macro_layer : float
        probability of mutating a macro layer

    grammar_path : str
        path to the grammar
    network_structure : list
        structure of the network
    network_structure_init : dict
        maximum number of layers when initialising
    levels_back : dict
        upper bound on the layers a given layer can receive as input
    max_epochs : int
        maximum number of epochs of the overall of the networks (can be
        used as a stop criteria of evolution)
    train_time : int
        maximum train time (in seconds) for each network
    macro_structure : list
        macro structure of the network
    output : str
        output layer of the network
    save_path : str
        path used to save the data generated during evolution
    """

    def __init__(self):
        self.random_seeds = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
        self.numpy_seeds = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
        self.num_generations = 200
        self._lambda = 4
        
        self.add_layer = 0.25
        self.reuse_layer = 0.15
        self.remove_layer = 0.25
        self.add_connection = 0.15
        self.remove_connection = 0.15
        self.dsge_layer = 0.15
        self.macro_layer = 0.3

        self.grammar_path = 'modules.grammar'
        self.network_structure = [('features', 30), ('classification', 10)]
        self.network_structure_init = {'features':[2,3,4], 'classification':[1]}
        self.levels_back = {'features': 5, 'classification': 1}
        self.max_epochs = 100000
        self.train_time = 10*60
        self.macro_structure = ['learning']
        self.output = 'softmax'
        self.save_path = './experiments/'


def save_pop(population, save_path, run, gen):
    """
        Save the population data
    """
    json_dump = []
    for ind in population:
        json_dump.append({'phenotype': ind.phenotype,
                          'fitness': ind.fitness,
                          'metrics': ind.metrics,
                          'trainable_parameters': ind.trainable_parameters,
                          'num_epochs': ind.num_epochs,
                          'time': ind.time})

    with open('%s/run_%d/gen_%d.csv' % (save_path, run, gen), 'w') as f_json:
        f_json.write(dumps(json_dump, indent=4))


def pickle_evaluator(evaluator, save_path, run):
    """
        Save the evaluator to later resume evolution
    """

    with open('%s/run_%d/evaluator.pkl' % (save_path, run), 'wb') as handle:
        pickle.dump(evaluator, handle, protocol=pickle.HIGHEST_PROTOCOL)


def pickle_population(population, save_path, run):
    """
        Save the population and random states to later resume evolution
    """
    with open('%s/run_%d/population.pkl' % (save_path, run), 'wb') as handle_pop:
        pickle.dump(population, handle_pop, protocol=pickle.HIGHEST_PROTOCOL)

    with open('%s/run_%d/random.pkl' % (save_path, run), 'wb') as handle_random:
        pickle.dump(random.getstate(), handle_random, protocol=pickle.HIGHEST_PROTOCOL)

    with open('%s/run_%d/numpy.pkl' % (save_path, run), 'wb') as handle_numpy:
        pickle.dump(np.random.get_state(), handle_numpy, protocol=pickle.HIGHEST_PROTOCOL)


def unpickle_population(save_path, run):
    """
        Load objects needed for resuming evolution
    """
    from glob import glob

    csvs = glob('%s/run_%d/*.csv' % (save_path, run))
    
    if csvs:
        csvs = [int(csv.split('/')[-1].replace('gen_','').replace('.csv','')) for csv in csvs]
        last_generation = max(csvs)

        with open('%s/run_%d/evaluator.pkl' % (save_path, run), 'rb') as handle_eval:
            pickle_evaluator = pickle.load(handle_eval)

        with open('%s/run_%d/population.pkl' % (save_path, run), 'rb') as handle_pop:
            pickle_population = pickle.load(handle_pop)

        pickle_population_fitness = [ind.fitness for ind in pickle_population]

        with open('%s/run_%d/random.pkl' % (save_path, run), 'rb') as handle_random:
            pickle_random = pickle.load(handle_random)

        with open('%s/run_%d/numpy.pkl' % (save_path, run), 'rb') as handle_numpy:
            pickle_numpy = pickle.load(handle_numpy)

        return last_generation, pickle_evaluator, pickle_population, pickle_population_fitness, pickle_random, pickle_numpy

    else:
        return None


def select_fittest(population, population_fits):
    """
        Seletion of the fittest individual by performance, and if there are individuals
        of equal performance the one with the least amount of trainable parameters is chosen
    """

    idx_max = np.argmax(population_fits)
    max_positions = np.where(np.array(population_fits) == population_fits[idx_max])[0]

    if len(max_positions) == 1:
        parent = population[max_positions[0]]

    else:
        max_positions = list(max_positions)
        num_parameters = [population[idx].trainable_parameters for idx in max_positions]
        parent = population[max_positions[np.argmin(num_parameters)]]

    return deepcopy(parent)


def mutation_dsge(layer, grammar, grammar_key):
    """
        DSGE mutations (check DSGE for futher details)
    """

    nt_keys = sorted(list(layer.keys()))
    nt_key = random.choice(nt_keys)
    nt_idx = random.randint(0, len(layer[nt_key])-1)

    sge_possibilities = []
    random_possibilities = []
    if len(grammar.grammar[nt_key]) > 1:
        sge_possibilities = list(set(range(len(grammar.grammar[nt_key]))) -\
                                 set([layer[nt_key][nt_idx]['ge']]))
        random_possibilities.append('ge')

    if layer[nt_key][nt_idx]['ga']:
        random_possibilities.extend(['ga', 'ga'])

    if random_possibilities:
        mt_type = random.choice(random_possibilities)

        if mt_type == 'ga':
            var_name = random.choice(sorted(list(layer[nt_key][nt_idx]['ga'].keys())))
            var_type, min_val, max_val, values = layer[nt_key][nt_idx]['ga'][var_name]
            value_idx = random.randint(0, len(values)-1)

            if var_type == 'int':
                new_val = random.randint(min_val, max_val)
            elif var_type == 'float':
                new_val = values[value_idx]+random.gauss(0, 0.15)
                new_val = np.clip(new_val, min_val, max_val)

            layer[nt_key][nt_idx]['ga'][var_name][-1][value_idx] = new_val

        elif mt_type == 'ge':
            layer[nt_key][nt_idx]['ge'] = random.choice(sge_possibilities)

        else:
            return NotImplementedError


def mutation(individual, grammar, add_layer, re_use_layer, remove_layer, add_connection, remove_connection, dsge_layer, macro_layer):
    """
        Network mutations: add and remove layer, add and remove connections, macro structure
    """

    ind = deepcopy(individual)
    
    for module in ind.modules:

        #add-layer (duplicate or new)
        for _ in range(random.randint(1,2)):
            if len(module.layers) < module.max_expansions and random.random() <= add_layer:
                if random.random() <= re_use_layer:
                    new_layer = random.choice(module.layers)
                else:
                    new_layer = grammar.initialise(module.module)

                insert_pos = random.randint(0, len(module.layers))

                #fix connections
                for _key_ in sorted(module.connections, reverse=True):
                    if _key_ >= insert_pos:
                        for value_idx, value in enumerate(module.connections[_key_]):
                            if value >= insert_pos-1:
                                module.connections[_key_][value_idx] += 1

                        module.connections[_key_+1] = module.connections.pop(_key_)


                module.layers.insert(insert_pos, new_layer)

                #make connections of the new layer
                if insert_pos == 0:
                    module.connections[insert_pos] = [-1]
                else:
                    connection_possibilities = list(range(max(0, insert_pos-module.levels_back), insert_pos-1))
                    if len(connection_possibilities) < module.levels_back-1:
                        connection_possibilities.append(-1)

                    sample_size = random.randint(0, len(connection_possibilities))
                    
                    module.connections[insert_pos] = [insert_pos-1] 
                    if sample_size > 0:
                        module.connections[insert_pos] += random.sample(connection_possibilities, sample_size)


        #remove-layer
        for _ in range(random.randint(1,2)):
            if len(module.layers) > 1 and random.random() <= remove_layer:
                remove_idx = random.randint(0, len(module.layers)-1)
                del module.layers[remove_idx]
                
                for _key_ in sorted(module.connections):
                    if _key_ > remove_idx:
                        if remove_idx in module.connections[_key_]:
                            module.connections[_key_].remove(remove_idx)

                        for value_idx, value in enumerate(module.connections[_key_]):
                            if value >= remove_idx:
                                module.connections[_key_][value_idx] -= 1
                        module.connections[_key_-1] = module.connections.pop(_key_)

                if remove_idx == 0:
                    module.connections[0] = [-1]


        for layer_idx, layer in enumerate(module.layers):
            if random.random() <= dsge_layer:
                mutation_dsge(layer, grammar, module.module)

            if layer_idx != 0 and random.random() <= add_connection:
                connection_possibilities = list(range(max(0, layer_idx-module.levels_back), layer_idx-1))
                connection_possibilities = list(set(connection_possibilities) - set(module.connections[layer_idx]))
                if len(connection_possibilities) > 0:
                    module.connections[layer_idx].append(random.choice(connection_possibilities))

            r_value = random.random()
            if layer_idx != 0 and r_value <= remove_connection:
                connection_possibilities = list(set(module.connections[layer_idx]) - set([layer_idx-1]))
                if len(connection_possibilities) > 0:
                    r_connection = random.choice(connection_possibilities)
                    module.connections[layer_idx].remove(r_connection)


    for macro_idx, macro in enumerate(ind.macro): 
        if random.random() <= macro_layer:
            mutation_dsge(macro, grammar, ind.macro_rules[macro_idx])
                    

    return ind


def get_total_epochs(save_path, run, last_gen):
    """
        Compute the total number of performed epochs (used to resume evolution)
    """

    from json import load

    total_epochs = 0
    for gen in range(0, last_gen+1):
        j = load(open('%s/run_%d/gen_%d.csv' % (save_path, run, gen)))
        num_epochs = [elm['num_epochs'] for elm in j]
        total_epochs += sum(num_epochs)

    return total_epochs


def main(run):
    """
        Evolutionary Strategy
    """

    config = Config()
    grammar = Grammar(config.grammar_path)
    cnn_eval = CNNEvaluator()

    unpickle = unpickle_population(config.save_path, run)

    if unpickle is None:
        makedirs('%s/run_%d/' % (config.save_path, run))
        random.seed(config.random_seeds[run])
        np.random.seed(config.numpy_seeds[run])
        cnn_eval = CNNEvaluator()
        pickle_evaluator(cnn_eval, config.save_path, run)
        last_gen = -1
        total_epochs = 0
	
    else:
        last_gen, cnn_eval, population, population_fits, pkl_random, pkl_numpy = unpickle
        random.setstate(pkl_random)
        np.random.set_state(pkl_numpy)
        total_epochs = get_total_epochs(config.save_path, run, last_gen)

    print last_gen

    for gen in range(last_gen+1, config.num_generations):

        if total_epochs is not None and total_epochs >= config.max_epochs:
            break

        print('[%d] %d' % (run, gen))
        if gen == 0:
            print('Creating initial population...')
            population = [Individual(config.network_structure, config.macro_structure, config.output).initialise(grammar, config.levels_back, config.network_structure_init)
                          for _ in range(config._lambda)]
            population_fits = [ind.evaluate(grammar, cnn_eval, config.train_time) for ind in population]
        else:
            parent = select_fittest(population, population_fits)
            offspring = [mutation(parent, grammar, config.add_layer,
                                  config.reuse_layer, config.remove_layer, 
                                  config.add_connection, config.remove_connection,
                                  config.dsge_layer, config.macro_layer) for _ in range(config._lambda)]
            population = [parent] + offspring
            population_fits = [ind.evaluate(grammar, cnn_eval, config.train_time) for ind in population]

        print('Best Fitness', max(population_fits))
        save_pop(population, config.save_path, run, gen)
        pickle_population(population, config.save_path, run)


if __name__ == '__main__':
    if len(argv) != 2:
        print('python multi_layer.py <run>')
        exit(-1)

    in_run = int(argv[1])

    main(in_run)


# Fast-DENSER++: Fast Deep Evolutionary Network Structured Representation

F-DENSER++ is a new extension to Deep Evolutionary Network Structured Evolution (DENSER). The vast majority of NeuroEvolution methods that optimise Deep Artificial Neural Networks (DANNs) only evaluate the candidate solutions for a fixed amount of epochs; this makes it difficult to effectively assess the learning strategy, and requires the best generated network to be further trained after evolution. F-DENSER++ enables the training time of the candidate solutions to grow continuously as necessary, i.e., in the initial generations the candidate solutions are trained for shorter times, and as generations proceed it is expected that longer training cycles enable better performances. Consequently, the models discovered by F-DENSER++ are fully-trained DANNs, and are ready for deployment after evolution, without the need for further training. 

```
@article{assunccao2019fast,
  title={Fast-DENSER++: Evolving Fully-Trained Deep Artificial Neural Networks},
  author={Assun{\c{c}}{\~a}o, Filipe and Louren{\c{c}}o, Nuno and Machado, Penousal and Ribeiro, Bernardete},
  journal={arXiv preprint arXiv:1905.02969},
  year={2019}
}

@article{assunccao2018denser,
  title={DENSER: deep evolutionary network structured representation},
  author={Assun{\c{c}}ao, Filipe and Louren{\c{c}}o, Nuno and Machado, Penousal and Ribeiro, Bernardete},
  journal={Genetic Programming and Evolvable Machines},
  pages={1--31},
  year={2018},
  publisher={Springer}
}
```

### Requirements
Currently this codebase only works with python 2. The following libraries are needed: tensorflow, keras, numpy, and sklearn, scipy, and jsmin. 

### Usage

`python f_denser.py -d <dataset> -c <config> -r <run> -g <grammar>`

-d [mandatory] can assume one of the following values: mnist, fashion-mnist, svhn, cifar10, cifar100-fine, cifar100-coarse, tiny-imagenet

-c [mandatory] is the path to a json configuration file. Check example/config.json for an example

-r [optional] the the run to be performed [0-14]

-g [mandatory] path to the grammar file to be used. Check example/modules.grammar for an example 

### Grammar

The mapping procedure of the available codebase supports production rules that can encode either topology or learning evolutionary units. The layers must start by \"layer:layer\_type\" where layer\_type indicates the type of the layer, e.g., conv (for convolutional), or fc (for fully-connected). To the moment the available layer types are convolutional (conv), pooling (pool-max or pool-avg), fully-connected (fc), dropout (dropout), and batch-normalization (batch-norm). The learning production rules must start by \"learning:algorithm\", where the algorithm can be gradient-descent, adam, or rmsprop. An example of a grammar can be found in examples/modules.grammar. 

The parameters are encoded in the production rules using the following format: [parameter-name, parameter-type, num-values, min-value, max-value], where the parameter-type can be integer or float; closed choice parameters are encoded using grammatical derivations. For each layer type the following parameters need to be defined:

|      Layer Type     |                                                       Parameters                                                       |
|:-------------------:|:----------------------------------------------------------------------------------------------------------------------:|
|    Convolutional    | Number of filters (num-filters), shape of the filters (filter-shape), stride, padding, activation function (act), bias |
|       Pooling       |                                       Kernel size (kernel-size), stride, padding                                       |
|   Fully-Connected   |                              Number of units (num-units), activation function (act), bias                              |
|       Dropout       |                                                          Rate                                                          |
| Batch-Normalization |                                                            -                                                           |

The current grammar example focuses on the simultaneous optimisation of the topoogy and learning. In case the user only intends to optimise the topology, the learning can be fixed by replacing the learning production rule by for example: \"<learning> ::= learning:gradient-descent lr:0.01 momentum:0.9 decay:0.0001 nesterov:True\". The same rationale applies for the topology.

The required parameters, and layers can be easily changed / extended by adapting the function that performs the mapping from the phenotype into a keras interpretable model. See the next section for further details.

### How to add new layers

To add new layers (or simply change the mandatory parameters) one needs to add (or adapt) the mapping from the phenotype to the keras interpretable model. This can be easily performed by adding the necessary code to the units.py file, in the \"assemble\_network\" function of the Evaluator class (starting in line 244). The code is to be added between the \"#Create layers -- ADD NEW LAYERS HERE\" and \"#END ADD NEW LAYERS\" comments. To change the parameters of an already existing layer there is just the need to change the call to the keras layer constructor. To add new layers a keras layer constructor must be added, and the parameters passed to it. For example, to add a Depthwise seperable 2D convolution we would write the following code:
```python
elif layer_type == 'sep_conv'
  sep_conv = keras.layers.SeparableConv2D(filters = int(layer_params['num-filters'][0]),
                      kernel_size = (int(layer_params['kernel-size'][0]), int(layer_params['kernel-size'][0])),
                      strides = (int(layer_params['stride'][0]), int(layer_params['stride'][0])),
                      padding = padding=layer_params['padding'][0],
                      dilation_rate = (int(layer_params['stride'][0]), int(layer_params['stride'][0])),
                      activation = layer_params['act'][0], 
                      use_bias = eval(layer_params['bias'][0]))
    layers.append(sep_conv)
```


### How to add new fitness functions

### Support

Any questions, comments or suggestion should be directed to Filipe Assunção ([fga@dei.uc.pt](mailto:fga@dei.uc.pt))
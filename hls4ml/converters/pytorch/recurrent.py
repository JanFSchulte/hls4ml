from hls4ml.converters.pytorch_to_hls import get_weights_data, pytorch_handler

rnn_layers = ['SimpleRNN', 'LSTM', 'GRU']


@pytorch_handler(*rnn_layers)
def parse_rnn_layer(operation, layer_name, input_names, input_shapes, node, class_object, data_reader, config):
    assert (operation in rnn_layers or operation == "RNN")

    layer = {}

    layer["name"] = layer_name

    layer['class_name'] = operation
    if operation == "RNN":
        layer['class_name'] = 'SimpleRNN'

    layer['return_sequences'] = False #parameter does not exist in pytorch
    layer['return_state'] = False #parameter does not exist in pytorch


    if layer['class_name'] == 'SimpleRNN' and "nonlinearity" in node.kwargs:
        layer['activation'] = node.kwargs["nonlinearity"] #GRU and LSTM are hard-coded to use tanh in pytorch 
    else:
        layer['activation'] = "tanh"

    print (node)

    layer['recurrent_activation'] = layer['activation'] #pytorch does not seem to differentiate between the two
    if layer['class_name'] == 'GRU':
        layer['recurrent_activation'] = 'sigmoid' #seems to be hard-coded in pytorch?

    if "batch_first" in node.kwargs:
        layer['time_major'] =  not node.kwargs["batch_first"]#
    else:
        layer['time_major'] = False
    # TODO Should we handle time_major?
    if layer['time_major']:
        raise Exception('hls4ml only supports "batch-first == True"')
    
    layer['n_timesteps'] = input_shapes[0][1]
    layer['n_in'] = input_shapes[0][2]

    if "hidden_size" in node.kwargs:
        layer['n_out'] = node.kwargs['hidden_size']
    else:    
        layer['n_out'] = input_shapes[1][-1]
    
    if "num_layers" in node.kwargs:
        raise Exception('hls4ml does not support num_layers > 1')

    if "bidirectional" in node.kwargs:
        if node.kwargs['bidirectional']:
            raise Exception('hls4ml does not support birectional RNNs')

    if "dropout" in node.kwargs:
        if node.kwarge["dropout"] > 0:
            raise Exception('hls4ml does not support RNNs with dropout')


    layer['weight_data'], layer['recurrent_weight_data'], layer['bias_data'], layer['recurrent_bias_data'] = get_weights_data(
        data_reader, layer['name'], ['weight_ih_l', 'weight_hh_l', 'bias_ih_l', 'bias_hh_l']
    )

    if layer['class_name'] == 'GRU':
        layer['apply_reset_gate'] = 'after' #Might be true for pytorch? It's not a free parameter

    output_shape = [[input_shapes[0][0], layer['n_timesteps'], layer['n_out']],[input_shapes[1][0], input_shapes[1][1],  ]]


    return layer, output_shape
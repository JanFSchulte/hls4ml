from hls4ml.converters.pytorch_to_hls import pytorch_handler

merge_layers = ['Add', 'Subtract', 'Multiply', 'Average', 'Maximum', 'Minimum', 'Cat', 'Dot']


@pytorch_handler(*merge_layers)
def parse_merge_layer(operation, layer_name, input_names, input_shapes, node, class_object, data_reader, config):
    assert operation in merge_layers

    layer = {}
    if operation == "Cat":
        layer['class_name'] = 'Concatenate'
    else:
        layer['class_name'] = operation
    layer['name'] = layer_name

    layer['op'] = operation

    if input_names is not None:
        layer['inputs'] = input_names

    output_shape = input_shapes[0][:]
    if layer['class_name'] == 'Concatenate':
        rank = len(input_shapes[0][1:])
        if rank > 3:
            raise Exception('ERROR: Concatenation of tensors with rank > 3 is not yet supported.')
        layer['op'] = layer['class_name'].lower() + f'{rank}d'
        layer['axis'] = int(node.args[1])
        output_shape[layer['axis']] += input_shapes[1][layer['axis']]
    elif layer['class_name'] == 'Dot':
        rank = len(input_shapes[0][1:])
        if rank > 1:
            raise Exception('ERROR: Dot of tensors with rank > 1 is not yet supported.')
        layer['op'] = layer['class_name'].lower() + f'{rank}d'
    else:
        layer['class_name'] = 'Merge'
    if len(layer['inputs']) > 2:
        raise Exception('ERROR: Merging more than two tensors is not yet supported.')

    return layer, output_shape


scatter_layers = ['scatter_add']


@pytorch_handler(*scatter_layers)
def parse_scatter_layer(operation, layer_name, input_names, input_shapes, node, class_object, data_reader, config):
    assert operation in scatter_layers
    assert len(input_shapes[0]) == len(input_shapes[1]) and len(input_shapes[0]) == len(input_shapes[2])

    layer = {}
    layer['name'] = layer_name
    layer['dim'] = node.args[1]

    if len(input_shapes[0]) == 2:
        layer['class_name'] = "ScatterAdd1D"
        layer['op'] = "ScatterAdd1D"
        layer['in_x'] = input_shapes[0][1]
        layer['index_x'] = input_shapes[1][1]
        layer['src_x'] = input_shapes[2][1]
    elif len(input_shapes[0]) == 3:
        layer['class_name'] = "ScatterAdd2D"
        layer['op'] = "ScatterAdd2D"
        layer['in_y'] = input_shapes[0][1]
        layer['in_x'] = input_shapes[0][2]
        layer['index_y'] = input_shapes[1][1]
        layer['index_x'] = input_shapes[1][2]
        layer['src_y'] = input_shapes[2][1]
        layer['src_x'] = input_shapes[2][2]
    elif len(input_shapes[0]) == 4:
        layer['class_name'] = "ScatterAdd3D"
        layer['op'] = "ScatterAdd3D"
        layer['in_z'] = input_shapes[0][1]
        layer['in_y'] = input_shapes[0][2]
        layer['in_x'] = input_shapes[0][3]
        layer['index_z'] = input_shapes[1][1]
        layer['index_y'] = input_shapes[1][2]
        layer['index_x'] = input_shapes[1][3]
        layer['src_z'] = input_shapes[2][1]
        layer['src_y'] = input_shapes[2][2]
        layer['src_x'] = input_shapes[2][3]
    else:
        raise Exception("ERROR: Unsupported number of dimensions for scatter_add")

    if input_names is not None:
        layer['inputs'] = input_names

    output_shape = input_shapes[0][:]
    return layer, output_shape

------------------------------------------------------------------
-- Export torch model config and layer param to disk.
------------------------------------------------------------------

require 'nn'
require 'xlua'
require 'json'
require 'paths'

npy4th = require 'npy4th';

torch.setdefaulttensortype('torch.FloatTensor')


PARAM_DIR = './param/'    -- Directory for saving layer param.
CONFIG_DIR = './config/'  -- Directory for saving net config.


---------------------------------------------------------------
-- Save layer param to disk.
-- Note we convert default FloatTensor to DoubleTensor
-- because of a weird bug in `npy4th`...
--
function save_param(save_name, weight, bias)
    npy4th.savenpy(PARAM_DIR..save_name..'.w.npy', weight:double())
    if bias then npy4th.savenpy(PARAM_DIR..save_name..'.b.npy', bias:double()) end
end

---------------------------------------------------------------
-- Save conv layer.
--
function conv_layer(layer, idx)
    local layer_name = 'conv'..idx
    save_param(layer_name, layer.weight, layer.bias)

    local nOutput = layer.nOutputPlane
    local kW,kH = layer.kW,layer.kH
    local dW,dH = layer.dW,layer.dH
    local pW,pH = layer.padW,layer.padH

    net_config[#net_config+1] = {
        ['id'] = #net_config,
        ['type'] = 'Convolution',
        ['name'] = layer_name,
        ['num_output'] = nOutput,
        ['kW'] = kW,
        ['kH'] = kH,
        ['dW'] = dW,
        ['dH'] = dH,
        ['pW'] = pW,
        ['pH'] = pH,
    }
end

---------------------------------------------------------------
-- Save bn runing_mean & running_var, and split weight & bias.
-- The reason for doing this is caffe uses BN+Scale to achieve
-- the full torch BN functionality.
--
function bn_layer(layer, idx)
    -- Save running_mean & running_var.
    local layer_name = 'bn'..idx
    save_param(layer_name, layer.running_mean, layer.running_var)

    local affine = layer.weight and true or false
    net_config[#net_config+1] = {
        ['id'] = #net_config,
        ['type'] = 'BatchNorm',
        ['name'] = layer_name,
        ['affine'] = affine,
    }

    -- If affine=true, save BN weight & bias as Scale layer param.
    if affine then
        layer_name = 'scale'..idx
        save_param(layer_name, layer.weight, layer.bias)

        net_config[#net_config+1] = {
            ['id'] = #net_config,
            ['type'] = 'Scale',
            ['name'] = layer_name,
        }
    end
end

---------------------------------------------------------------
-- Save pooling layer.
--
function pooling_layer(layer, idx)
    local layer_name = 'pool'..idx

    local pool_type = torch.type(layer)=='nn.SpatialMaxPooling' and 0 or 1
    local kW,kH = layer.kW,layer.kH
    local dW,dH = layer.dW,layer.dH
    local pW,pH = layer.padW,layer.padH

    net_config[#net_config+1] = {
        ['id'] = #net_config,
        ['type'] = 'Pooling',
        ['name'] = layer_name,
        ['pool_type'] = pool_type,
        ['kW'] = kW,
        ['kH'] = kH,
        ['dW'] = dW,
        ['dH'] = dH,
        ['pW'] = pW,
        ['pH'] = pH,
    }
end

---------------------------------------------------------------
-- Save linear layer.
--
function linear_layer(layer, idx)
    local layer_name = 'linear'..idx
    save_param(layer_name, layer.weight, layer.bias)

    local nOutput = layer.weight:size(1)
    net_config[#net_config+1] = {
        ['id'] = #net_config,
        ['type'] = 'InnerProduct',
        ['name'] = layer_name,
        ['num_output'] = nOutput,
    }
end

---------------------------------------------------------------
-- Save dropout layer.
--
function dropout_layer(layer, idx)
    local p = layer.p
    net_config[#net_config+1] = {
        ['id'] = #net_config,
        ['type'] = 'Dropout',
        ['name'] = layer_name,
        ['dropout_ratio'] = p,
    }
end

---------------------------------------------------------------
-- Save layers with no param.
--
function noparam_layer(layer, idx)
    -- Map torch layer type to caffe layer type and layer name.
    local layer_type_name = {
        ['nn.ReLU'] = {'ReLU', 'relu'..idx},
        ['nn.View'] = {'Flatten', 'flatten'..idx},
        ['nn.SoftMax'] = {'Softmax', 'softmax'..idx},
    }

    local type_and_name = layer_type_name[torch.type(layer)]

    net_config[#net_config+1] = {
        ['id'] = #net_config,
        ['type'] = type_and_name[1],
        ['name'] = type_and_name[2],
    }
end


if #arg ~= 5 then
    print('Usage: th torch/export.lua [path_to_torch_model] [input_shape]')
    print('e.g. th torch/export.lua ./net.t7 {1,1,28,28}')
    return
else
    net_path = arg[1]
    input_shape = { tonumber(arg[2]), tonumber(arg[3]),
                    tonumber(arg[4]), tonumber(arg[5]) }
end

paths.mkdir(PARAM_DIR)
paths.mkdir(CONFIG_DIR)

net = torch.load(net_path)
net_config = {}

-- Add input layer config.
net_config[#net_config+1] = {
    ['id'] = #net_config,
    ['type'] = 'DummyData',
    ['name'] = 'data',
    ['input_shape'] = input_shape,
}

-- Map layer type to it's saving function.
layerfn = {
    ['nn.SpatialConvolution'] = conv_layer,
    ['nn.SpatialBatchNormalization'] = bn_layer,
    ['nn.SpatialMaxPooling'] = pooling_layer,
    ['nn.SpatialAveragePooling'] = pooling_layer,
    ['nn.Linear'] = linear_layer,
    ['nn.Dropout'] = dropout_layer,
    ['nn.ReLU'] = noparam_layer,
    ['nn.View'] = noparam_layer,
    ['nn.SoftMax'] = noparam_layer,
}

print('==> Exporting..')
for i = 1,#net do
    local layer = net:get(i)
    local layer_type = torch.type(layer)
    print(string.format('... Layer %d : %s', i, layer_type))

    local save_layer = layerfn[layer_type]
    assert(save_layer, 'ERROR: save '..layer_type..' not supported yet!')
    save_layer(layer, i)
end

-- Save config file.
json.save(CONFIG_DIR..'net.json', net_config)

-- Graph.
graph = torch.zeros(#net_config, #net_config)

-- TODO: build graph from net structure.
-- For now just sequential.
for i = 1, graph:size(1) do
    graph[i][i] = 1
    if i < graph:size(1) then
        graph[i][i+1] = 1
    end
end
npy4th.savenpy(CONFIG_DIR..'graph.npy', graph)

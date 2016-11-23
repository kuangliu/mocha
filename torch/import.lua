------------------------------------------------------------------
-- Rebuild torch model from saved layer param and config file.
------------------------------------------------------------------

require 'nn'
require 'xlua'
require 'json'
require 'paths'

npy4th = require 'npy4th'

torch.setdefaulttensortype('torch.FloatTensor')


PARAM_DIR = './param/'    -- Directory for saving layer param.
CONFIG_DIR = './config/'  -- Directory for saving net config.


--------------------------------------------------------
-- Load saved param from .npy file.
--
function load_param(layer_name)
    assert(paths.dirp(PARAM_DIR), 'ERROR: '..PARAM_DIR..' not exist!')
    -- Weight is compulsive.
    local weight = npy4th.loadnpy(PARAM_DIR..layer_name..'.w.npy')
    -- Bias is optional.
    local bias_path = PARAM_DIR..layer_name..'.b.npy'
    local bias = paths.filep(bias_path) and npy4th.loadnpy(bias_path) or nil
    return weight, bias
end

--------------------------------------------------------
-- New linear layer.
--
function linear_layer(layer_config)
    local weight, bias = load_param(layer_config.name)
    local inputSize = weight:size(2)
    local outputSize = weight:size(1)
    local layer = nn.Linear(inputSize, outputSize)
    layer.weight:copy(weight)
    layer.bias:copy(bias)
    return layer
end

--------------------------------------------------------
-- New conv layer.
--
function conv_layer(layer_config)
    local weight, bias = load_param(layer_config.name)
    local nInputPlane = weight:size(2)
    local nOutputPlane = weight:size(1)
    local kW = layer_config.kW
    local kH = layer_config.kH
    local dW = layer_config.dW
    local dH = layer_config.dH
    local pW = layer_config.pW
    local pH = layer_config.pH
    local layer = nn.SpatialConvolution(nInputPlane, nOutputPlane, kW,kH,dW,dH,pW,pH)
    layer.weight:copy(weight)
    if bias then
        layer.bias:copy(bias)
    else
        layer:noBias()
    end
    return layer
end

--------------------------------------------------------
-- New bn layer.
-- If BatchNorm is followed by Scale in caffemodel,
--   affine=true.
-- If BatchNorm with no Scale, affine=false.
--
function bn_layer(layer_config)
    local running_mean, running_var = load_param(layer_config.name)
    local nOutput = running_mean:size(1)
    local affine = layer_config.affine
    local layer = nn.SpatialBatchNormalization(nOutput, nil, nil, affine)
    layer.running_mean:copy(running_mean)
    layer.running_var:copy(running_var)
    return layer
end

--------------------------------------------------------
-- New scale layer.
-- 1. If the previous layer is BN, merge the weight/bias.
-- 2. If not... TODO
--
function scale_layer(layer_config)
    local weight, bias = load_param(layer_config.name)
    local lastbn = net:get(#net)
    assert(torch.type(lastbn) == 'nn.SpatialBatchNormalization',
                'ERROR: Scale must follow BatchNorm.')
    assert(lastbn.affine, 'ERROR: Scale layer must follow BatchNorm.')
    lastbn.weight:copy(weight)
    lastbn.bias:copy(bias)
end

--------------------------------------------------------
-- New pooling layer.
--
function pooling_layer(layer_config)
    local pooling_type = layer_config.pool_type  -- Max or average.
    local kW = layer_config.kW
    local kH = layer_config.kH
    local dW = layer_config.dW
    local dH = layer_config.dH
    local pW = layer_config.pW
    local pH = layer_config.pH

    local layer
    if pooling_type == 0 then
        layer = nn.SpatialMaxPooling(kW,kH,dW,dH,pW,pH):ceil()
    elseif pooling_type == 1 then
        layer = nn.SpatialAveragePooling(kW,kH,dW,dH,pW,pH):ceil()
    else
        error('ERROR: pooling type not supported!')
    end
    return layer
end

--------------------------------------------------------
-- New ReLU layer.
--
function relu_layer()
    return nn.ReLU(true)
end

--------------------------------------------------------
-- New flatten layer.
--
function flatten_layer()
    return nn.View(-1)
end

--------------------------------------------------------
-- New dropout layer.
--
function dropout_layer(layer_config)
    local p = layer_config.dropout_ratio
    return nn.Dropout(p)
end

--------------------------------------------------------
-- New softmax layer.
--
function softmax_layer()
    return nn.SoftMax()
end


-- Map layer_type to building function.
layerfn = {
    ['Convolution'] = conv_layer,
    ['BatchNorm'] = bn_layer,
    ['Scale'] = scale_layer,
    ['ReLU'] = relu_layer,
    ['Pooling'] = pooling_layer,
    ['Flatten'] = flatten_layer,
    ['InnerProduct'] = linear_layer,
    ['Dropout'] = dropout_layer,
    ['Softmax'] = softmax_layer,
}

-- Load net config.
config_path = CONFIG_DIR..'net.json'
assert(paths.filep(config_path), 'ERROR: '..config_path..' not exist!')
net_config = json.load(config_path)

print('==> Importing..')

net = nn.Sequential()  -- Transfer saved param to net.
flattened = false      -- Need flatten before adding any linear layers.

for i = 2,#net_config do  -- Skip input layer.
    layer_config = net_config[i]

    local layer_type = layer_config.type
    print(string.format('... Layer %d : %s', i-1, layer_type))

    -- If not flattened, add a flatten layer before any linear layers.
    if not flattened and layer_type == 'InnerProduct' then
        net:add(nn.View(-1))
        flattened = true
    end

    -- Contains flatten layer, then no need to add.
    if layer_type == 'Flatten' then flattened = true end

    -- Add a new layer.
    local getlayer = layerfn[layer_type]
    if not getlayer then
        error('ERROR: '..layer_type..' not supported yet!')
    end

    local layer = getlayer(layer_config)
    if layer then net:add(layer) end    -- Scale layer returns nil.
end

print(net)
torch.save('net.t7', net)

-- test
print('Testing..')
net:evaluate()
x = torch.randn(1,3,96,96)
npy4th.savenpy('x.npy',x)

y = net:float():forward(x:float())
print(y)
torch.save('y_torch.t7', y)

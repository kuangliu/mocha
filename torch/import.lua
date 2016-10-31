------------------------------------------------------------------
-- Rebuild torch model from saved layer params and config file.
------------------------------------------------------------------

require 'nn';
require 'xlua';
require 'paths';
npy4th = require 'npy4th';

torch.setdefaulttensortype('torch.FloatTensor')


-- Directory containing saved layer params.
PARAM_DIR = './params/'


--------------------------------------------------------
-- Load saved params from .npy file.
--
function load_params(layer_name)
    assert(paths.dirp(PARAM_DIR), 'ERROR '..PARAM_DIR..' not exist!')
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
function linear_layer(layer_name)
    -- Load params.
    local weight, bias = load_params(layer_name)
    -- Define linear layer.
    local inputSize = weight:size(2)
    local outputSize = weight:size(1)
    local layer = nn.Linear(inputSize, outputSize)
    -- Copy params.
    layer.weight:copy(weight)
    layer.bias:copy(bias)
    return layer
end

--------------------------------------------------------
-- New conv layer.
--
function conv_layer(layer_name)
    -- Load params.
    local weight, bias = load_params(layer_name)
    -- Define conv layer.
    local nInputPlane = weight:size(2)
    local nOutputPlane = weight:size(1)
    local kW,kH = tonumber(splited[5]),tonumber(splited[6])
    local dW,dH = tonumber(splited[7]),tonumber(splited[8])
    local pW,pH = tonumber(splited[9]),tonumber(splited[10])
    local layer = nn.SpatialConvolution(nInputPlane, nOutputPlane, kW,kH,dW,dH,pW,pH)
    -- Copy params.
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
-- If BatchNorm followed by Scale in caffemodel, affine=true.
-- If BatchNorm with no Scale, affine=false.
--
function bn_layer(layer_name)
    -- Load params.
    local running_mean, running_var = load_params(layer_name)
    -- Define BN layer.
    local nOutput = running_mean:size(1)
    -- Default assume [BN-Scale] in caffemodel, which affine=true.
    local layer = nn.SpatialBatchNormalization(nOutput, nil, nil, true)
    -- Copy params.
    layer.running_mean:copy(running_mean)
    layer.running_var:copy(running_var)
    return layer
end

--------------------------------------------------------
-- New scale layer.
-- 1. If the previous layer is BN, merge the weight/bias.
-- 2. If not... TODO
--
function scale_layer(layer_name)
    local weight, bias = load_params(layer_name)
    local lastbn = net:get(#net)
    assert(torch.type(lastbn) == 'nn.SpatialBatchNormalization',
                'Scale must follow BatchNorm.')
    lastbn.weight:copy(weight)
    lastbn.bias:copy(bias)
end

--------------------------------------------------------
-- New pooling layer.
--
function pooling_layer(layer_name)
    local pooling_type = splited[4] -- Max or average.
    local kW,kH = tonumber(splited[5]),tonumber(splited[6])
    local dW,dH = tonumber(splited[7]),tonumber(splited[8])
    local pW,pH = tonumber(splited[9]),tonumber(splited[10])

    local layer
    if pooling_type == '0' then
        layer = nn.SpatialMaxPooling(kW,kH,dW,dH,pW,pH):ceil()
    elseif pooling_type == '1' then
        layer = nn.SpatialAveragePooling(kW,kH,dW,dH,pW,pH):ceil()
    else
        error('ERROR pooling type not supported!')
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
function dropout_layer()
    local p = tonumber(splited[4])
    return nn.Dropout(p)
end

--------------------------------------------------------
-- New softmax layer.
--
function softmax_layer()
    return nn.SoftMax()
end


-- Map layer_type to it's processing function.
layerfn = {
    Convolution = conv_layer,
    BatchNorm = bn_layer,
    Scale = scale_layer,
    ReLU = relu_layer,
    Pooling = pooling_layer,
    Flatten = flatten_layer,
    InnerProduct = linear_layer,
    Dropout = dropout_layer,
    Softmax = softmax_layer,
}

-- Config file path.
cfgpath = PARAM_DIR..'net.config'
assert(paths.filep(cfgpath), 'ERROR '..cfgpath..' not exist!')

-- Transfer saved params to net.
net = nn.Sequential()

-- Need flatten before adding any linear layers.
flattened = false

print('==> importing..')
for line in io.lines(cfgpath) do
    splited = string.split(line, '\t')
    local layer_type = splited[2]
    local layer_name = splited[3]

    i = (i or 0) + 1
    print('... layer '..i..': '..layer_type)

    -- If not flattened, add a flatten layer before any linear layers.
    if not flattened and layer_type == 'InnerProduct' then
        net:add(nn.View(-1))
        flattened = true
    end

    -- Contains flatten layer, no need to automatically add it.
    if layer_type == 'Flatten' then flattened = true end

    -- Add a new layer.
    local getlayer = layerfn[layer_type]
    if not getlayer then
        error('[ERROR]'..layer_type..' not supported yet!')
    end

    local layer = getlayer(layer_name)
    if layer then net:add(layer) end    -- Scale layer returns nil.
end

print(net)
torch.save('net.t7', net)

-- test
print('testing..')
net:evaluate()
x = torch.randn(1,1,28,28)
npy4th.savenpy('x.npy',x)

y = net:float():forward(x:float())
print(y)
torch.save('y_torch.t7', y)

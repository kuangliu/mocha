------------------------------------------------------------------
-- Reconstruct a Torch model from saved layer params and log file.
------------------------------------------------------------------
require 'nn';
require 'xlua';
require 'paths';
npy4th = require 'npy4th';

torch.setdefaulttensortype('torch.FloatTensor')

--------------------------------------------------------
-- Load saved weight & bias as .npy file.
--
function load_params(layer_name)
    local param_dir = './params/'
    assert(paths.dirp(param_dir), param_dir..' Not Exist!')
    local weight = npy4th.loadnpy(param_dir..layer_name..'_weight.npy')
    local bias = npy4th.loadnpy(param_dir..layer_name..'_bias.npy')
    return weight, bias
end

--------------------------------------------------------
-- New linear layer
--
function linear_layer(layer_name)
    -- load params
    local weight, bias = load_params(layer_name)
    -- define linear layer
    local inputSize = weight:size(2)
    local outputSize = weight:size(1)
    local layer = nn.Linear(inputSize, outputSize)
    -- copy params
    layer.weight:copy(weight)
    layer.bias:copy(bias)
    return layer
end

--------------------------------------------------------
-- New conv layer
--
function conv_layer(layer_name)
    -- load params
    local weight, bias = load_params(layer_name)
    -- define conv layer
    local nInputPlane = weight:size(2)
    local nOutputPlane = weight:size(1)
    local kW,kH = tonumber(sp[4]),tonumber(sp[5])
    local dW,dH = tonumber(sp[6]),tonumber(sp[7])
    local pW,pH = tonumber(sp[8]),tonumber(sp[9])
    local layer = nn.SpatialConvolution(nInputPlane, nOutputPlane, kW,kH,dW,dH,pW,pH)
    -- copy params
    layer.weight:copy(weight)
    layer.bias:copy(bias)
    return layer
end

--------------------------------------------------------
-- New bn layer
--
function bn_layer(layer_name)
    -- load params
    local param_dir = './params/'
    local running_mean = npy4th.loadnpy(param_dir..layer_name..'_mean.npy')
    local running_var = npy4th.loadnpy(param_dir..layer_name..'_var.npy')
    -- deine BN layer
    local nOutput = running_mean:size(1)
    local layer = nn.SpatialBatchNormalization(nOutput, nil, nil, false) -- No affine
    -- copy params
    layer.running_mean:copy(running_mean)
    layer.running_var:copy(running_var)
    return layer
end


-- get layers from log
logfile = io.open('./params/net.log')

-- transfer saved params to net
net = nn.Sequential()
print('importing..')
while true do
    line = logfile:read('*l')
    if not line then break end

    sp = string.split(line, '\t')
    layer_type = sp[2]
    layer_name = sp[3]

    i = (i or 0) + 1
    print('==> layer '..i..': '..layer_type)
    if layer_type == 'Linear' then
        net:add(linear_layer(layer_name))
    elseif layer_type == 'ReLU' then
        net:add(nn.ReLU(true))
    elseif layer_type == 'Flatten' then
        net:add(nn.View(-1))
    elseif layer_type == 'Convolution' then
        net:add(conv_layer(layer_name))
    elseif layer_type == 'BatchNorm' then
        net:add(bn_layer(layer_name))
    else
        print('[ERROR]'..layer_type..' not supported yet!')
    end
end

torch.save('net.t7', net)

-- test
print('testing..')
net:evaluate()
x = torch.randn(1,2,10,10)
y = net:float():forward(x:float())
print(y)

npy4th.savenpy('x.npy', x)

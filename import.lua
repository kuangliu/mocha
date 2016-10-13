------------------------------------------------------------------
-- Reconstruct a Torch model from saved layer params and log file.
------------------------------------------------------------------
require 'nn';
require 'xlua';
npy4th = require 'npy4th';

torch.setdefaulttensortype('torch.FloatTensor')

param_dir = './params/'
logfile = io.open(param_dir..'net.log')

-- transfer params to net
net = nn.Sequential()

-- loop all layers
print('importing..')
while true do
    line = logfile:read('*l')
    if not line then break end

    sp = string.split(line, '\t')
    layer_type = sp[2]
    layer_name = sp[3]

    i = (i or 0) + 1
    if layer_type == 'Linear' then
        print('==> layer '..i..': '..layer_type)
        -- load saved params
        layer_weight = npy4th.loadnpy(param_dir..layer_name..'_weight.npy')
        layer_bias = npy4th.loadnpy(param_dir..layer_name..'_bias.npy')
        -- define Linear layer
        inputSize = layer_weight:size(2)
        outputSize = layer_weight:size(1)
        layer = nn.Linear(inputSize, outputSize)
        -- copy params
        layer.weight:copy(layer_weight)
        layer.bias:copy(layer_bias)
        net:add(layer)
    elseif layer_type == 'ReLU' then
        print('==> layer '..i..': '..layer_type)
        net:add(nn.ReLU(true))
    elseif layer_type == 'Flatten' then
        print('==> layer '..i..': '..layer_type)
        net:add(nn.View(-1))
    elseif layer_type == 'Convolution' then
        print('==> layer '..i..': '..layer_type)
        layer_weight = npy4th.loadnpy(param_dir..layer_name..'_weight.npy')
        layer_bias = npy4th.loadnpy(param_dir..layer_name..'_bias.npy')
        -- define Conv layer
        nInputPlane = layer_weight:size(2)
        nOutputPlane = layer_weight:size(1)
        kW,kH = tonumber(sp[4]),tonumber(sp[5])
        dW,dH = tonumber(sp[6]),tonumber(sp[7])
        pW,pH = tonumber(sp[8]),tonumber(sp[9])
        layer = nn.SpatialConvolution(nInputPlane, nOutputPlane, kW,kH,dW,dH,pW,pH)
        -- copy params
        layer.weight:copy(layer_weight)
        layer.bias:copy(layer_bias)
        net:add(layer)
    end
end

torch.save('net.t7', net)


-- test
print('testing..')
x = torch.randn(1,1,5,5)
y = net:float():forward(x:float())
print(y)

npy4th.savenpy('x.npy', x)

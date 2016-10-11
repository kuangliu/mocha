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

    splited = string.split(line, '\t')
    layer_type = splited[2]
    layer_name = splited[3]

    if layer_type == 'Linear' then
        -- load saved params
        layer_weight = npy4th.loadnpy(param_dir..layer_name..'_weight.npy')
        layer_bias = npy4th.loadnpy(param_dir..layer_name..'_bias.npy')
        -- build a new layer
        inputSize = layer_weight:size(2)
        outputSize = layer_weight:size(1)
        layer = nn.Linear(inputSize, outputSize)
        -- copy params
        layer.weight:copy(layer_weight)
        layer.bias:copy(layer_bias)
        net:add(layer)

    elseif layer_type == 'ReLU' then
        net:add(nn.ReLU(true))

    end

    -- print info
    i = (i or 0) + 1
    print('==> layer '..i..': '..layer_type)
end

torch.save('test.t7', net)

n1 = torch.load('./test.t7')
n2 = torch.load('./model/net.t7')

x = torch.randn(2048)

n1:forward(x:float())
n2:forward(x)

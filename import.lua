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
while true do
    line = logfile:read('*l')
    if not line then break end

    splited = string.split(line, '\t')
    layer_type = splited[2]
    layer_name = splited[3]

    if layer_type == 'InnerProduct' then   -- Linear
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
    end
end

torch.save('test.t7', net)



-- n1 = torch.load('./test.t7')
-- n2 = torch.load('./model/net.t7')
--
-- x = torch.randn(10)
--
-- n1:forward(x)
-- n2:forward(x)

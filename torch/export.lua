------------------------------------------------------------------
-- Export layer params of a torch model to disk.
------------------------------------------------------------------

require 'nn';
require 'xlua';
require 'paths';
npy4th = require 'npy4th';

torch.setdefaulttensortype('torch.FloatTensor')

---------------------------------------------------------------
-- Save weight & bias to disk as save_name.w/b.npy
--
function save_param(save_name, weight, bias)
    npy4th.savenpy(save_dir..save_name..'.w.npy', weight)
    if bias then npy4th.savenpy(save_dir..save_name..'.b.npy', bias) end
end

---------------------------------------------------------------
-- Write layer info to log file.
--
function logging(idx, layer_type, layer_name, cfg)
    local s = idx..'\t'..layer_type..'\t'..layer_name

    cfg = cfg or {}
    for _,v in pairs(cfg) do
        s = s .. '\t' .. v
    end
    logfile:write(s..'\n')
end

---------------------------------------------------------------
-- Save conv layer params.
--
function conv_layer(layer, idx)
    local layer_name = 'conv'..idx
    save_param(layer_name, layer.weight, layer.bias)

    local nOutput = layer.nOutputPlane
    local kW,kH = layer.kW,layer.kH
    local dW,dH = layer.dW,layer.dH
    local pW,pH = layer.padW,layer.padH
    local cfg = {nOutput, kW,kH,dW,dH,pW,pH}
    logging(idx, 'Convolution', layer_name, cfg)
end

---------------------------------------------------------------
-- Save an runing_mean&running_var, and split weight & bias out.
-- The reason for doing this is caffe uses BN+Scale to achieve
-- the full torch BN functionality.
--
function bn_layer(layer, idx)
    -- save running_mean & running_var as bn.w/b.npy
    local layer_name = 'bn'..idx
    save_param(layer_name, layer.running_mean, layer.running_var)
    logging(idx, 'BatchNorm', layer_name)

    -- save weight & bias as scale.w/b.npy
    layer_name = 'scale'..idx
    save_param(layer_name, layer.weight, layer.bias)
    logging(idx, 'Scale', layer_name)
end

---------------------------------------------------------------
-- Logging pooling layer configs.
--
function pooling_layer(layer, idx)
    local layer_name = 'pool'..idx

    local pool_type = torch.type(layer)=='nn.SpatialMaxPooling' and 0 or 1
    local kW,kH = layer.kW,layer.kH
    local dW,dH = layer.dW,layer.dH
    local pW,pH = layer.padW,layer.padH
    local cfg = {pool_type, kW,kH,dW,dH,pW,pH}
    logging(idx, 'Pooling', layer_name, cfg)
end

---------------------------------------------------------------
-- Save linear layer params.
--
function linear_layer(layer, idx)
    local layer_name = 'linear'..idx
    save_param(layer_name, layer.weight, layer.bias)

    local nOutput = layer.weight:size(1)
    logging(idx, 'InnerProduct', layer_name, {nOutput})
end

---------------------------------------------------------------
-- For layer has no param or config, just logging.
--
function noparam_layer(layer, idx)
    local layer_name
    if torch.type(layer) == 'nn.ReLU' then
        layer_name = 'relu'..idx
        logging(idx, 'ReLU', layer_name)
    elseif torch.type(layer) == 'nn.View' then
        layer_name = 'flatten'..idx
        logging(idx, 'Flatten', layer_name)
    elseif torch.type(layer) == 'nn.SoftMax' then
        layer_name = 'softmax'..idx
        logging(idx, 'Softmax', layer_name)
    end
end

save_dir = './params/'
paths.mkdir(save_dir)

-- load torch model
net = torch.load('./net.t7')

logfile = io.open(save_dir..'net.log', 'w')

-- map layer type to it's saving function
layerfunc = {
    ['nn.SpatialConvolution'] = conv_layer,
    ['nn.SpatialBatchNormalization'] = bn_layer,
    ['nn.SpatialMaxPooling'] = pooling_layer,
    ['nn.SpatialAveragePooling'] = pooling_layer,
    ['nn.Linear'] = linear_layer,
    ['nn.ReLU'] = noparam_layer,
    ['nn.View'] = noparam_layer,
    ['nn.SoftMax'] = noparam_layer,
}

for i = 1,#net do
    layer = net:get(i)
    layer_type = torch.type(layer)

    save_layer = layerfunc[layer_type]
    if not save_layer then
        error('[ERROR] Save '..layer_type..' not supported yet!')
    end
    save_layer(layer, i)
end

logfile:close()

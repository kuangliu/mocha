------------------------------------------------------------------
-- Export torch model layer params to disk.
------------------------------------------------------------------

require 'nn';
require 'xlua';
require 'paths';

npy4th = require 'npy4th';

torch.setdefaulttensortype('torch.FloatTensor')


-- Directory for saving layer params.
SAVE_DIR = './params/'


---------------------------------------------------------------
-- Save layer params to disk.
--
function save_param(save_name, weight, bias)
    npy4th.savenpy(SAVE_DIR..save_name..'.w.npy', weight)
    if bias then npy4th.savenpy(SAVE_DIR..save_name..'.b.npy', bias) end
end

---------------------------------------------------------------
-- Write layer config to config file.
--
function logging(idx, layer_type, layer_name, cfg)
    local s = idx..'\t'..layer_type..'\t'..layer_name

    cfg = cfg or {}
    for _,v in pairs(cfg) do
        s = s .. '\t' .. v
    end
    cfgfile:write(s..'\n')
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
-- Save bn runing_mean&running_var, and split weight & bias out.
-- The reason for doing this is caffe uses BN+Scale to achieve
-- the full torch BN functionality.
--
function bn_layer(layer, idx)
    -- Save running_mean & running_var.
    local layer_name = 'bn'..idx
    save_param(layer_name, layer.running_mean, layer.running_var)
    logging(idx, 'BatchNorm', layer_name)

    -- Save weight & bias.
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


paths.mkdir(SAVE_DIR)

-- Load torch model.
net = torch.load('./net.t7')

cfgfile = io.open(SAVE_DIR..'net.config', 'w')

-- Map layer type to it's saving function.
layerfn = {
    ['nn.SpatialConvolution'] = conv_layer,
    ['nn.SpatialBatchNormalization'] = bn_layer,
    ['nn.SpatialMaxPooling'] = pooling_layer,
    ['nn.SpatialAveragePooling'] = pooling_layer,
    ['nn.Linear'] = linear_layer,
    ['nn.ReLU'] = noparam_layer,
    ['nn.View'] = noparam_layer,
    ['nn.SoftMax'] = noparam_layer,
}

print('==> Exporting..')
for i = 1,#net do
    local layer = net:get(i)
    local layer_type = torch.type(layer)
    print('... '..'Layer '..i..' : '..layer_type)

    local save_layer = layerfn[layer_type]
    assert(save_layer, 'ERROR save '..layer_type..' not supported yet!')
    save_layer(layer, i)
end

cfgfile:close()

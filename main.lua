require 'nn';

-- input: 1x1x5x5
net = nn.Sequential()
net:add(nn.SpatialConvolution(1,2,3,3));
net:add(nn.SpatialBatchNormalization(2,nil,nil,false));
net:add(nn.ReLU(true))
net:add(nn.View(-1));
net:add(nn.Linear(18,10))

torch.save('./model/net.t7', net)


-- x = torch.randn(1,1,5,5)
-- y = net:forward(x)

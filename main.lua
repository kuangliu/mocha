require 'nn';

-- input: 1x1x100x100
net = nn.Sequential()
net:add(nn.SpatialConvolution(1,2,3,3))
net:add(nn.View(18))
net:add(nn.Linear(18,10))

torch.save('./model/net.t7', net)

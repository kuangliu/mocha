require 'nn';

-- input: 1x1x10x10
net = nn.Sequential()
net:add(nn.SpatialConvolution(2,3,3,3))
net:add(nn.SpatialBatchNormalization(3,nil,nil,false))
net:add(nn.ReLU(true))
net:add(nn.SpatialMaxPooling(2,2,1,1,0,0))
net:add(nn.SpatialConvolution(3,4,3,3))
net:add(nn.SpatialBatchNormalization(4,nil,nil,false))
net:add(nn.ReLU(true))
net:add(nn.SpatialAveragePooling(2,2,1,1,0,0))
net:add(nn.View(-1))
net:add(nn.Linear(64,10))

torch.save('./model/net.t7', net)

x = torch.randn(1,2,10,10)
y = net:forward(x)
print(#y)

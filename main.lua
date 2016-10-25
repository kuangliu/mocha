require 'nn';
npy4th = require 'npy4th';

net = torch.load('./net.t7')
net

net:remove(7)
net:remove(7)
net:remove(7)
net:remove(7)
net:remove(7)
net:remove(6)
net

net:add(nn.SpatialAveragePooling(4,4,1,1))

x=torch.randn(1,1,28,28)
y = net:float():forward(x:float())
#y

torch.save('./net.t7', net)

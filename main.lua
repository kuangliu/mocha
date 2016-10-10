require 'nn';

net = nn.Sequential()
net:add(nn.Linear(10,2))

torch.save('net.t7', net)

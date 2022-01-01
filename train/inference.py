from network_building import MyModel
import torch

learning_rate = 1e-5
model = MyModel()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
checkpoint = torch.load('model_weights_opt1.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']
model.train()
print('Loading model complete')

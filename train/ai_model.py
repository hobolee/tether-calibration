import rospy
import numpy as np
import os
import torch
from network_building import MyModel


# path to model
model_path = 'model_weights_opt_mac.pth'
learning_rate = 1e-5
model = MyModel()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']
model.train()
print('Loading model complete')


def main():
    rospy.init_node('ai_model', anonymous=True)
    global pub, back
    # publish
    pub = rospy.Publisher('node_name', data, queue_size=1)


import torch
import matplotlib.pyplot as plt

loss = torch.load('loss')
print(loss)

plt.plot(loss.float())
plt.show()
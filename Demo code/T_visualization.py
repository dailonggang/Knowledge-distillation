import numpy as np
import matplotlib.pyplot as plt


logits = np.array([-5, 2, 7, 9])
softmax1 = np.exp(logits) / sum(np.exp(logits))
print(softmax1)

plt.plot(softmax1, label='T=1')
T = 2
softmax2 = np.exp(logits/T) / sum(np.exp(logits/T))
plt.plot(softmax2, label='T=2')
T = 5
softmax3 = np.exp(logits/T) / sum(np.exp(logits/T))
plt.plot(softmax3, label='T=5')
T = 10
softmax4 = np.exp(logits/T) / sum(np.exp(logits/T))
plt.plot(softmax4, label='T=10')
T = 100
softmax5 = np.exp(logits/T) / sum(np.exp(logits/T))
plt.plot(softmax5, label='T=100')

plt.xticks(np.arange(4), ['cat', 'dog', 'pig', 'mouse'])
plt.ylabel('score')
plt.legend()
plt.show()


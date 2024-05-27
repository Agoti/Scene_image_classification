
import matplotlib.pyplot as plt
import numpy as np
import os

# Load the loss and accuracy
path = 'log.txt'
with open(path, 'r') as f:
    train_acc = []
    loss = []
    val_acc = []
    lines = f.readlines()
    for line in lines:
        if line.startswith('Epoch'):
            loss.append(float(line.split(' ')[3].strip()))
            val_acc.append(float(line.split(' ')[6].strip()))
            if 'Train' in line:
                train_acc.append(float(line.split(' ')[9].strip()))
    
# Plot loss and acc in the same figure. Use the left y-axis for loss and the right y-axis for acc
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(loss, 'r-')
ax2.plot(val_acc, 'b-')
if len(train_acc) > 0:
    ax2.plot(train_acc, 'g-')

# Set the labels
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss', color='r')
ax2.set_ylabel('Accuracy', color='b')

# Save the figure
plt.show()



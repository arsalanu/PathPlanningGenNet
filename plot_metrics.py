import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import json
import statistics as stats

matplotlib.use('TkAgg')
matplotlib.rcParams['figure.figsize'] = (22, 10)

# This file can be used to load a written training record and plot as needed.

with open('training_record.json') as jp:
    data = json.load(jp)

epochs = 15

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

plt.ylim(50, 100)

plt.subplot(2,4,1)
plt.ylabel('Accuracy', size=8)
plt.xlabel('Epoch', size=8)
plt.title('Training accuracy', size=10)
plt.plot(range(epochs), data['training_accuracy'], "-m", linewidth=0.9)
plt.grid()

plt.subplot(2,4,2)
plt.ylabel('Loss', size=8)
plt.xlabel('Epoch', size=8)
plt.title('Training loss', size=10)
plt.plot(range(epochs), data['training_loss'], "-c",linewidth=0.9)
plt.grid()

plt.subplot(2,4,3)
plt.ylabel('Accuracy', size=8)
plt.xlabel('Epoch', size=8)
plt.title('Validation accuracy', size=10)
plt.plot(range(epochs), data['validation_accuracy'][::3], "-r", linewidth=0.9)
plt.grid()

plt.subplot(2,4,4)
plt.ylabel('Loss', size=8)
plt.xlabel('Epoch', size=8)
plt.title('Validation loss', size=10)
plt.plot(range(epochs), data['validation_loss'][::3], "-g", linewidth=0.9)
plt.grid()


plt.subplot(2,4,5)
plt.ylabel('\%', size=8)
plt.xlabel('Epoch', size=8)
plt.title('Validity (\%)', size=10)
plt.grid()

c1 = data['performance_metrics']['validity']['0.4']
c2 = data['performance_metrics']['validity']['0.5']
c3 = data['performance_metrics']['validity']['0.6']
mean = [stats.mean([c1[i], c2[i], c3[i]]) for i in range(len(c1))]

plt.plot(range(epochs), data['performance_metrics']['validity']['0.4'], "-r", linewidth=0.75, label='$C > 0.4$')
plt.plot(range(epochs), data['performance_metrics']['validity']['0.5'], "-g", linewidth=0.75, label='$C > 0.5$')
plt.plot(range(epochs), data['performance_metrics']['validity']['0.6'], "-b", linewidth=0.75, label='$C > 0.6$')
plt.plot(range(epochs), mean, "-.k", linewidth=0.75, label='Mean')
plt.legend(loc="upper right")


plt.subplot(2,4,6)
plt.ylabel('\%', size=8)
plt.xlabel('Epoch', size=8)
plt.title('Optimality (\%)', size=10)
plt.grid()

c1 = data['performance_metrics']['optimality']['0.4']
c2 = data['performance_metrics']['optimality']['0.5']
c3 = data['performance_metrics']['optimality']['0.6']
mean = [stats.mean([c1[i], c2[i], c3[i]]) for i in range(len(c1))]

plt.plot(range(epochs), data['performance_metrics']['optimality']['0.4'], "-r", linewidth=0.75, label='$C > 0.4$')
plt.plot(range(epochs), data['performance_metrics']['optimality']['0.5'], "-g", linewidth=0.75, label='$C > 0.5$')
plt.plot(range(epochs), data['performance_metrics']['optimality']['0.6'], "-b", linewidth=0.75, label='$C > 0.6$')
plt.plot(range(epochs), mean, "-.k", linewidth=0.75, label='Mean')
plt.legend(loc="upper right")


plt.subplot(2,4,7)
plt.ylabel('\%', size=8)
plt.xlabel('Epoch', size=8)
plt.title('Search space reduction (\%)', size=10)
plt.grid()

c1 = data['performance_metrics']['search_space']['0.4']
c2 = data['performance_metrics']['search_space']['0.5']
c3 = data['performance_metrics']['search_space']['0.6']
mean = [stats.mean([c1[i], c2[i], c3[i]]) for i in range(len(c1))]

plt.plot(range(epochs), data['performance_metrics']['search_space']['0.4'], "-r", linewidth=0.75, label='$C > 0.4$')
plt.plot(range(epochs), data['performance_metrics']['search_space']['0.5'], "-g", linewidth=0.75, label='$C > 0.5$')
plt.plot(range(epochs), data['performance_metrics']['search_space']['0.6'], "-b", linewidth=0.75, label='$C > 0.6$')
plt.plot(range(epochs), mean, "-.k", linewidth=0.75, label='Mean')
plt.legend(loc="upper right")


# plt.savefig('PATHGEN_NET/training_graphs.png')
plt.show()
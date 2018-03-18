import matplotlib.pyplot as plt

with open('results.txt', 'r') as f:
    results = []
    line = f.readline()
    while line:
        results.append(int(line))
        line = f.readline()

plt.plot(results, 'bo')
plt.show()

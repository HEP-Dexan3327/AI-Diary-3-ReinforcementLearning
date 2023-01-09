from matplotlib import pyplot as plt
from cmdargs import args
import numpy as np
import pandas as pd

scores = {}
epochs = {}
highTiles = {}
rolling = []

with open(f'{args.file}', 'r') as f:
    data = f.read()
    for line in data.split('\n'):
        s = line.split()
        if len(s) > 3:
            scores.update({int(s[0]):int(s[2])})
            epochs.update({int(s[0]):int(s[1])})
            highTiles.update({int(s[0]): int(s[3])})
scores = dict(sorted(scores.items(), key=lambda k: k[0]))
epochs = dict(sorted(epochs.items(), key=lambda k: k[0]))
highTiles = dict(sorted(highTiles.items(), key=lambda k: k[0]))

MARK = 50

df = pd.DataFrame({'epoch': scores.keys(), 'score': scores.values(),
                   'num': epochs.values(), 'hi': highTiles.values()})
df['rolling'] = df['score'].rolling(MARK).mean()
df['rolling_epoch'] = df['num'].rolling(MARK).mean()
df['rolling_high'] = df['hi'].rolling(MARK).mean()

fig, axs = plt.subplots(2)
axs[0].plot(np.arange(len(scores)), df['rolling'])
axs[1].plot(np.arange(len(scores)), df['rolling_high'])
plt.xlabel('epochs')
plt.ylabel('score')
plt.show()

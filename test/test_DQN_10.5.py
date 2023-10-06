import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets

REPO_NAME = 'FlowerDockingSensorimotorLoopRL'
REPO_PATH = os.path.abspath(__file__)
while os.path.basename(REPO_PATH) != REPO_NAME: REPO_PATH = os.path.dirname(REPO_PATH)
if REPO_PATH not in sys.path: sys.path.append(REPO_PATH)

from Gym.ReactiveSingleFlowerDocking import ModelTrainer


def test1():
    # set DIR to date string
    DIR = 'saved_models_' + time.strftime("%m.%d", time.localtime())
    agent, returns = ModelTrainer.train_from_random(DIR)
    fig, ax = plt.subplots(dpi=300)
    ax.plot(returns)
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Average Return')
    fig.savefig(os.path.join(DIR, 'returns.png'))
    plt.show()
    plt.close(fig)

def main():
    return test1()

if __name__ == '__main__':
    main()
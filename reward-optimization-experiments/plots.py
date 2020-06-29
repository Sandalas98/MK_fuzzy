import matplotlib.pyplot as plt
import pandas as pd

def print_steps(m, title):
    df = pd.DataFrame(m)
    df.set_index('trial', inplace=True)

    fig, ax = plt.subplots()
    df['steps_in_trial'].plot(ax=ax)
    ax.set_title(title)
    ax.set_xlabel('Trial')
    plt.show()

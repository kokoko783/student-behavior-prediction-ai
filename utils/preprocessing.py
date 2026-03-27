import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def visualize_dataset(file):
    df = pd.read_csv(file)
    sns.pairplot(df, hue='engagement')
    plt.show()
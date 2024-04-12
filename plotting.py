import matplotlib.pyplot as plt

def plot_class_distribution(annotations):
    # Getting the class distribution
    class_dist = annotations.groupby(['ClassLabel'])['Length'].mean()

    # Plotting the distribution
    fig, ax = plt.subplots(num='Class Distribution')
    fig.suptitle('Class Distribution')
    # ax.set_title('Class Distribution', y=1.08)
    ax.pie(class_dist, labels=class_dist.index, autopct='%1.1f%%', shadow=False, startangle=90)
    ax.axis('equal')
    plt.show()
    annotations.reset_index(inplace=True)
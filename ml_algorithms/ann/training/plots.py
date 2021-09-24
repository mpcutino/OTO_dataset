import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd


def plot_acc_history(history, acc_keys, s_path, colors, title="Accuracy per epoch"):
    plt.title(title)
    for i, acc_key in enumerate(acc_keys):
        x = range(1, len(history[acc_key]) + 1)
        plt.plot(x, history[acc_key], linestyle='-', color=colors[i], label=acc_key)
    plt.legend()
    plt.savefig(s_path)
    plt.show()
    plt.close()


def plot_loss_history(history, loss_keys, s_path, colors):
    plot_acc_history(history, loss_keys, s_path, colors, title="Loss per epoch")


def plot_confusion_matrix(confusion_matrix, class_names, img_name="confusion_matrix.png"):
    df_cm = pd.DataFrame(confusion_matrix, class_names, class_names)

    plt.figure(figsize=(10, 7))
    sn.set(font_scale=0.8)  # for label size
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 9}, fmt='d')  # font size

    plt.savefig(img_name)
    # plt.show()
    plt.close()

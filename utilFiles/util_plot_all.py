import os

import matplotlib.pyplot as plt
import warnings; warnings.filterwarnings("ignore")


import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16})
def plot_functions_1d_np(target_x, target_y, context_x, context_y, pred_y, var, it=0, save_to_dir="", save_img=True):
    plt.rcParams.update({'font.size': 22})

    plt.clf()
    plt.plot(target_x[0], pred_y[0], "b", linewidth=2.5, label = "Model Prediction")
    plt.plot(target_x[0], target_y[0], "k:", linewidth=2, label = "True Function")
    plt.plot(context_x[0], context_y[0], 'ko', markersize=8)
    plt.fill_between(
        target_x[0, :, 0],
        pred_y[0, :, 0] - var[0, :, 0],
        pred_y[0, :, 0] + var[0, :, 0],
        alpha=0.7,
        facecolor='#65c999',
        interpolate=True,
        label = "Variance"
    )
    plt.xlabel("X value")
    plt.ylabel("Y value")
    # plt.xlim(-1.5,1.5)
    # plt.ylim(-1.5,1.5)
    # plt.ylim(-3,7)
    # plt.ylim(-3,5)
    # plt.ylim(-0.7,1.5)

    # plot details
    plt.grid(False)
    plt.legend(loc='upper right')
    # plt.show()
    if save_img:
        plt.savefig(f"{save_to_dir}/plotv5{it}" + ".png", format='png',dpi=300,bbox_inches='tight')
        # plt.show()
    else:
        plt.show()

def plot_functions_alea_ep_1d(target_x, target_y, context_x, context_y, pred_y, epis,alea,save_img=True,save_to_dir="eval_images", save_name="a.png"):
    plt.rcParams.update({'font.size': 22})
    plt.clf()
    plt.plot(target_x[0], target_y[0], "k:", linewidth=2.6, label = "True Function")
    plt.plot(target_x[0], pred_y[0], "b", linewidth=2, label = "Prediction")
    plt.plot(context_x[0], context_y[0], 'ko', markersize=8)
    plt.vlines(x=5.0, ymin=-4, ymax=8, linestyles='--')
    # plt.title(r"$\lambda_1 = 0.1, \lambda_2 = 1.0$")
    plt.title(r"ENP-C")
    plt.fill_between(
        target_x[0, :, 0],
        pred_y[0, :, 0] - epis[0, :, 0],
        pred_y[0, :, 0] + epis[0, :, 0],
        alpha=0.7,
        facecolor='#65c999',
        interpolate=True,
        label = "Epistemic Unc.",
    )
    # plt.fill_between(
    #     target_x[0, :, 0],
    #     pred_y[0, :, 0] - alea[0, :, 0],
    #     pred_y[0, :, 0] + alea[0, :, 0],
    #     alpha=0.2,
    #     facecolor='red',
    #     interpolate=True,
    #     label = "Aleatoric",
    # )

    plt.xlabel("X value")
    plt.ylabel("Y value")
    plt.xlim(-5,10)
    plt.ylim(-5,12)

    # plt.xlim(-2,3)
    # plt.ylim(-3,3)
    # plt.ylim(-0.6,1.0)
    # plt.ylim(-0.7,1.5)

    # plot details
    plt.grid(False)
    # plt.legend(loc=2, bbox_to_anchor=(1,1))
    plt.legend()#loc='upper left')
    # plt.show()
    if not os.path.exists(save_to_dir):
        os.mkdir(save_to_dir)
    if save_img:
        plt.savefig(f"{save_to_dir}/Al5Ep{save_name}" + ".png", format='png',dpi=300,bbox_inches='tight')
        # plt.show()
    else:
        plt.show()

def plot_functions_var_1d(target_x, target_y, context_x, context_y, pred_y, var,save_img=True,save_to_dir="eval_images", save_name="a.png"):
    plt.rcParams.update({'font.size': 22})
    plt.clf()
    plt.plot(target_x[0], target_y[0], "k:", linewidth=2.6, label = "True Function")
    plt.plot(target_x[0], pred_y[0], "b", linewidth=2, label = "Prediction")
    plt.plot(context_x[0], context_y[0], 'ko', markersize=8)
    plt.vlines(x=5.0, ymin=-4, ymax=8, linestyles='--')
    plt.title("CNP Model")
    plt.fill_between(
        target_x[0, :, 0],
        pred_y[0, :, 0] - var[0, :, 0],
        pred_y[0, :, 0] + var[0, :, 0],
        alpha=0.7,
        facecolor='#65c999',
        interpolate=True,
        label = "Variance",
    )
    # plt.fill_between(
    #     target_x[0, :, 0],
    #     pred_y[0, :, 0] - alea[0, :, 0],
    #     pred_y[0, :, 0] + alea[0, :, 0],
    #     alpha=0.2,
    #     facecolor='red',
    #     interpolate=True,
    #     label = "Aleatoric",
    # )

    plt.xlabel("X value")
    plt.ylabel("Y value")
    plt.xlim(-5,10)
    plt.ylim(-5,12)

    # plt.xlim(-2,3)
    # plt.ylim(-3,3)
    # plt.ylim(-0.6,1.0)
    # plt.ylim(-0.7,1.5)

    # plot details
    plt.grid(False)
    # plt.legend(loc=2, bbox_to_anchor=(1,1))
    plt.legend()#loc='upper left')
    # plt.show()
    if not os.path.exists(save_to_dir):
        os.mkdir(save_to_dir)
    if save_img:
        plt.savefig(f"{save_to_dir}/Al5Ep{save_name}" + ".png", format='png',dpi=300,bbox_inches='tight')
        # plt.show()
    else:
        plt.show()


def plot_functions_multiple(target_x, target_y, context_x, context_y, predictions, epis=0,alea=0,save_img=True,save_to_dir="eval_images", save_name="a.png"):
    plt.rcParams.update({'font.size': 22})
    plt.clf()
    plt.plot(target_x[0], target_y[0], "k:", linewidth=1.6, label = "True Function")
    for index, pred_y in enumerate(predictions):
        if index == 0:
            plt.plot(target_x[0], pred_y[0], "b", linewidth=1, label="Predictions")
        else:
            plt.plot(target_x[0], pred_y[0], "b", linewidth=1)
    plt.plot(context_x[0], context_y[0], 'ko', markersize=8)
    print("min: ", min(min(target_y)))
    plt.ylim(( min(min(target_y))-0.5, max(max(target_y))+1.0))
    # plt.vlines(x=5.0, ymin=-5, ymax=5)
    # plt.fill_between(
    #     target_x[0, :, 0],
    #     pred_y[0, :, 0] - epis[0, :, 0],
    #     pred_y[0, :, 0] + epis[0, :, 0],
    #     alpha=0.7,
    #     facecolor='#65c999',
    #     interpolate=True,
    #     label = "Epistemic Uncertainty",
    # )
    # plt.fill_between(
    #     target_x[0, :, 0],
    #     pred_y[0, :, 0] - alea[0, :, 0],
    #     pred_y[0, :, 0] + alea[0, :, 0],
    #     alpha=0.2,
    #     facecolor='red',
    #     interpolate=True,
    #     label = "Aleatoric",
    # )

    plt.xlabel("X value")
    plt.ylabel("Y value")
    # plt.xlim(-1.5,1.5)
    # plt.ylim(-0.75,1.2)
    # plt.ylim(-0.6,1.0)
    # plt.ylim(-3,5)

    # plot details
    plt.grid(False)
    plt.legend(loc="upper right")
    # plt.show()
    if not os.path.exists(save_to_dir):
        os.mkdir(save_to_dir)
    if save_img:
        plt.savefig(f"{save_to_dir}/a{save_name}" + ".png", format='png',dpi=300,bbox_inches='tight')
        # plt.show()
    else:
        plt.show()


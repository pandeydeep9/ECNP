import matplotlib.pyplot as plt
import numpy as np
import pylab as p
import torch.nn.functional as F
def plot_functions_3d(input,mean, variance,mask,it,location = "",save=False, w=32,h=32,c=3):
    image_width = w
    image_height = h
    channels = c
    plt.clf()
    num_images = len(input)

    titles = ["Image", "CM", "Prediction", "Variance"]
    num_rows = len(titles)
    max_num_cols = 5
    fig, axs = plt.subplots(min(5, max(num_images, 2)), num_rows)
    for a_ind, a in enumerate(axs):
        for b_ind, b in enumerate(a):
            if a_ind == 0:
                b.set_title(titles[b_ind])
            b.axes.xaxis.set_visible(False)
            b.axes.yaxis.set_visible(False)

    for index, inp in enumerate(input):
        if index >=max_num_cols: break

        # print("inp shape: ", inp.shape)
        inp = inp.transpose(0,1)
        inp = inp.transpose(1,2)
        input_image = inp.detach().cpu().numpy().reshape(image_width,image_height,channels)

        axs[index,0].imshow((input_image*1.0))

        mask_untiled = mask[index].detach().cpu().numpy().reshape(image_width,image_height,1)
        # mask_used = np.repeat(mask_untiled, channels, axis = -1)
        # print(type(mask_untiled), type(input_image), input_image.shape)
        mask_used = np.tile(mask_untiled, (1,1,channels))
        # print(type(mask_used), type(mask_used), mask_used.shape)

        mask_show = mask[index].detach().cpu().numpy().reshape(image_width,image_height)
        axs[index,1].imshow(mask_used*1.0)
        # context_image = mask_used * input_image
        # axs[index, 2].imshow(context_image*1.0)
        mean_pred = F.relu(mean[index]).detach().cpu().numpy().reshape(image_width,image_height,channels)
        axs[index, 2].imshow((mean_pred*1.0))
        variance_pred = variance[index].detach().cpu().numpy().reshape(image_width,image_height,channels)
        axs[index, 3].imshow((variance_pred*1.0))

    plt.suptitle(f"Iteration: {it}")
    if save:
        plt.savefig(f"{location}plot{it}" + ".png")
    else:
        plt.show()

from pathlib import Path
def plot_functions_mnist(input,mean, variance,mask,it,location = "",save=False,w=28,h=28,c=1):
    Path(location).mkdir(parents=True, exist_ok=True)
    image_width = 28
    image_height = 28
    channels = 1
    plt.clf()
    num_images = len(input)
    titles = ["Image", "Mask", "Context", "Prediction", "Aleatoric"]
    num_rows = 5
    max_num_cols = 5
    fig, axs = plt.subplots(min(5, max(num_images,2) ), num_rows)
    for a_ind, a in enumerate(axs):
        for b_ind, b in enumerate(a):
            if a_ind == 0:
                b.set_title(titles[b_ind])
            b.axes.xaxis.set_visible(False)
            b.axes.yaxis.set_visible(False)

    for index, inp in enumerate(input):
        if index >=max_num_cols: break

        input_image = inp.detach().cpu().numpy().reshape(28,28)
        axs[index,0].imshow(input_image*1.0)

        mask_used = mask[index].detach().cpu().numpy().reshape(28,28)
        axs[index,1].imshow(mask_used*1.0)
        context_image = mask_used * input_image
        axs[index, 2].imshow(context_image*1.0)
        mean_pred = F.relu(mean[index]).detach().cpu().numpy().reshape(28,28)
        axs[index, 3].imshow(mean_pred*1.0)
        variance_pred = variance[index].detach().cpu().numpy().reshape(28,28)
        axs[index, 4].imshow(variance_pred*1.0)

    plt.suptitle(f"Iteration: {it}")
    plt.grid(False)
    if save:
        plt.savefig(f"{location}/plot{it}" + ".png")
    else:
        plt.show()

import torch
def plot_functions_mnist_new(input,mean, v, alpha, beta, target_y, cm,it,location = "",save=False,w=28,h=28,c=1):
    Path(location).mkdir(parents=True, exist_ok=True)
    image_width = 28
    image_height = 28
    channels = 1
    plt.clf()
    num_images = len(input)
    num_rows = 6
    max_num_cols = 5
    context = torch.zeros(target_y.shape)
    context = cm


    alea_pred = beta / (alpha - 1)
    variance_pred = alea_pred / v
    titles = ["Image", "Context Mask", "Context Set", "Mean Pr.", "Var. Pr.", "Alea. Pr."]
    fig, axs = plt.subplots(min(max_num_cols, max(num_images,2) ), num_rows)
    for a_ind, a in enumerate(axs):
        for b_ind, b in enumerate(a):

            if a_ind == 0:
                b.set_title(titles[b_ind])
            b.axes.xaxis.set_visible(False)
            b.axes.yaxis.set_visible(False)

    for index, inp in enumerate(input):
        if index >=max_num_cols: break
        input_image = inp.detach().cpu().numpy().reshape(28,28)
        axs[index,0].imshow(input_image)
        mask_used = context[index].detach().cpu().numpy().reshape(28,28)

        axs[index,1].imshow(mask_used)
        context_image = mask_used * input_image

        axs[index, 2].imshow(context_image)
        mean_pred = mean[index].detach().cpu().numpy().reshape(28,28)

        axs[index, 3].imshow(mean_pred)

        var_pred = variance_pred[index].detach().cpu().numpy().reshape(28,28)
        axs[index, 4].imshow(var_pred)

        al_pred = alea_pred[index].detach().cpu().numpy().reshape(28,28)
        axs[index, 5].imshow(al_pred)

    plt.suptitle(f"Iteration: {it}")
    plt.grid(False)
    if save:
        plt.savefig(f"{location}/plot{it}" + ".png")
        plt.close()
    else:
        plt.show()
        plt.close()


def plot_functions_3d_edl(input, mean, v, alpha, beta, target_y, context_mask, it, location="", save=False, w=32, h=32,
                                 c=3):
    Path(location).mkdir(parents=True, exist_ok=True)
    image_width = w
    image_height = h
    channels = c
    plt.clf()
    num_images = len(input)

    alea_pred = beta / (alpha - 1)
    variance_pred = alea_pred / v

    num_cols = 4
    titles = ["Image", "CM",  "Prediction", "Epistemic", "Aleatoric"]
    max_num_rows = 5
    fig, axs = plt.subplots( nrows=min(max_num_rows, max(num_images, 2)),ncols=num_cols)
    for a_ind, a in enumerate(axs):
        for b_ind, b in enumerate(a):
            if a_ind == 0:
                b.set_title(titles[b_ind])
            b.axes.xaxis.set_visible(False)
            b.axes.yaxis.set_visible(False)

    for index, inp in enumerate(input):
        if index >=max_num_rows: break

        # print("inp shape: ", inp.shape)
        inp = inp.transpose(0,1)
        inp = inp.transpose(1,2)
        input_image = inp.detach().cpu().numpy().reshape(image_width,image_height,channels)

        axs[index,0].imshow((input_image*1.0))

        mask_untiled = context_mask[index].detach().cpu().numpy().reshape(image_width,image_height,1)
        # mask_used = np.repeat(mask_untiled, channels, axis = -1)
        # print(type(mask_untiled), type(input_image), input_image.shape)
        mask_used = np.tile(mask_untiled, (1,1,channels))
        # print(type(mask_used), type(mask_used), mask_used.shape)

        mask_show = context_mask[index].detach().cpu().numpy().reshape(image_width,image_height)
        axs[index,1].imshow(mask_show)
        # context_image = mask_used * input_image
        # axs[index, 2].imshow((context_image*1.0))
        mean_pred = F.relu(mean[index]).detach().cpu().numpy().reshape(image_width,image_height,channels)
        # print(mean_pred)

        axs[index, 2].imshow((mean_pred*1.0), vmin = 0,vmax=1.0)
        # variance_pred[index] = variance_pred[index]/torch.max(variance_pred[index])
        variance = variance_pred[index].detach().cpu().numpy().reshape(image_width,image_height,channels)*1.0
        variance = np.mean(variance, axis=-1)
        axsi3 = axs[index, 3]
        varee = axsi3.imshow(variance, cmap='plasma')
        fig.colorbar(varee, ax = axsi3)
        # print(variance)

        # alea_pred[index] = alea_pred[index]/torch.max(alea_pred[index])
        # alea = alea_pred[index].detach().cpu().numpy().reshape(image_width,image_height,channels)
        # axs[index, 4].imshow((alea*1.0))

    plt.suptitle(f"Iteration: {it}")
    # plt.show()
    if save:
        plt.savefig(f"{location}/plot{it}" + ".png")
        # plt.show()
    else:

        plt.show()


def plot_functions_3d_ep_alea(input, mean, var,epis,alea, target_y, context_mask, it, location="", save=False, w=32, h=32,
                                 c=3):
    Path(location).mkdir(parents=True, exist_ok=True)
    image_width = w
    image_height = h
    channels = c
    plt.clf()
    num_images = len(input)

    alea_pred = alea
    variance_pred = epis

    num_rows = 6
    titles = ["Image", "CM", "Context Set", "Mean Pr.", "Epis. Pr.", "Alea. Pr."]
    max_num_cols = 5
    fig, axs = plt.subplots(min(max_num_cols, max(num_images, 2)), num_rows)



    for a_ind, a in enumerate(axs):
        for b_ind, b in enumerate(a):
            if a_ind == 0:
                b.set_title(titles[b_ind])
            b.axes.xaxis.set_visible(False)
            b.axes.yaxis.set_visible(False)

    for index, inp in enumerate(input):
        if index >=max_num_cols: break

        # print("inp shape: ", inp.shape)
        inp = inp.transpose(0,1)
        inp = inp.transpose(1,2)
        input_image = inp.detach().cpu().numpy().reshape(image_width,image_height,channels)

        axs[index,0].imshow((input_image*1.0))

        mask_untiled = context_mask[index].detach().cpu().numpy().reshape(image_width,image_height,1)
        # mask_used = np.repeat(mask_untiled, channels, axis = -1)
        # print(type(mask_untiled), type(input_image), input_image.shape)
        mask_used = np.tile(mask_untiled, (1,1,channels))
        # print(type(mask_used), type(mask_used), mask_used.shape)

        mask_show = context_mask[index].detach().cpu().numpy().reshape(image_width,image_height)
        axs[index,1].imshow(mask_used)
        # print("mask used: ",mask_used.shape)

        non_observed = 0.8 * (mask_used==0)
        non_observed[:,:,:2] = 0


        context_image = input_image*mask_used + non_observed
        axs[index, 2].imshow((context_image*1.0))
        mean_pred = F.relu(mean[index]).detach().cpu().numpy().reshape(image_width,image_height,channels)
        # print(mean_pred)

        axs[index, 3].imshow((mean_pred*1.0), vmin = 0,vmax=1.0)
        variance_pred[index] = variance_pred[index]/torch.max(variance_pred[index])
        variance = variance_pred[index].detach().cpu().numpy().reshape(image_width,image_height,channels)
        axs[index, 4].imshow((variance*1.0))
        # print(variance)

        alea_pred[index] = alea_pred[index]/torch.max(alea_pred[index])
        alea = alea_pred[index].detach().cpu().numpy().reshape(image_width,image_height,channels)
        axs[index, 5].imshow((alea*1.0))

    plt.suptitle(f"Iteration: {it}")
    if save:
        plt.savefig(f"{location}plot{it}" + ".png")
        # plt.show()
    else:
        plt.show()

import torch
import random
from models.image_completion_helpers import the_image_grid, make_context_mask


def get_context_target_2d(image_batch, num_ctx_pts = -1, is_Test_Time = False):
    bs, c, h, w = image_batch.size()
    device = image_batch.device

    if num_ctx_pts == -1:
        num_context_points = random.randint(10, 100)
    else:
        num_context_points = num_ctx_pts

    num_target_points = h*w-num_context_points#500-num_context_points#random.randint(100, 200)
    if is_Test_Time:
        #If test, then all image points are targetoints
        num_target_points = h*w-num_context_points

    context_mask = torch.zeros(bs, h, w).bool().to(device)
    target_mask = torch.zeros(bs, h, w).bool().to(device)
    for i in range(bs):
        context_mask_one = torch.zeros(h * w).bool()
        target_mask_one = torch.zeros(h * w).bool()

        all_idx = torch.tensor(
            random.sample(range(0, h * w), num_context_points + num_target_points)).to(device)
        context_idx = all_idx[:num_context_points]
        context_mask_one[context_idx] = 1
        target_mask_one[all_idx] = 1

        context_mask_one = context_mask_one.view(h, w)
        target_mask_one = target_mask_one.view(h, w)

        context_mask[i] = context_mask_one
        target_mask[i] = target_mask_one

    all_xy_pos = the_image_grid(h, w).to(device)

    context_x_wo_mask = all_xy_pos.view(h * w, -1).unsqueeze(0).expand(bs, -1, -1)
    context_y_wo_mask = image_batch.view(bs, c, h * w).transpose(1, 2)#*255

    context_mask_cor = context_mask.unsqueeze(1).repeat(1, c, 1, 1)  # repeat for each channel
    num_context = torch.nonzero(context_mask[0]).size(0)
    context_nonzero_idx = torch.nonzero(context_mask)
    context_x = context_nonzero_idx[:, 1:].view(bs, num_context, 2).float() / h
    context_y = image_batch[context_mask_cor].view(bs, c, num_context_points)

    target_mask_cor = target_mask.unsqueeze(1).repeat(1, c, 1, 1)  # repeat for each channel
    num_target = torch.nonzero(target_mask[0]).size(0)
    target_nonzero_idx = torch.nonzero(target_mask)
    target_x = target_nonzero_idx[:, 1:].view(bs, num_target, 2).float() / h
    target_y = image_batch[target_mask_cor].view(bs, c, num_target_points+num_context_points)

    # context_y = context_y.permute(0, 2, 1)#q*255
    # TODO: transpose
    context_y = context_y.transpose( 2, 1)#q*255
    target_y = target_y.transpose(2,1)
    # print("context mask: ", context_mask)

    # return ((context_x, context_y), context_x_wo_mask), context_y_wo_mask, context_mask
    return ((context_x, context_y), target_x), target_y, context_mask, (context_x_wo_mask,context_y_wo_mask)

def get_context_target_for_plot_single(image_batch, num_ctx_pts = -1, all_idx= None):
    c, h, w = image_batch.size()
    device = image_batch.device

    if num_ctx_pts == -1 or all_idx is None:
        raise NotImplementedError
    else:
        num_context_points = num_ctx_pts

    num_target_points = h*w-num_context_points


    context_mask_one = torch.zeros(h * w).bool().to(device)
    target_mask_one = torch.ones(h * w).bool().to(device)

    context_idx = all_idx[:num_context_points]
    context_mask_one[context_idx] = 1
    # target_mask_one[all_idx] = 1

    context_mask = context_mask_one.view(h, w)
    target_mask = target_mask_one.view(h, w)

    all_xy_pos = the_image_grid(h, w).to(device)

    context_x_wo_mask = all_xy_pos.view(h * w, -1)
    context_y_wo_mask = image_batch.view( c, h * w).transpose(0, 1)#*255

    context_mask_cor = context_mask.unsqueeze(0).repeat(c, 1, 1)  # repeat for each channel
    num_context = torch.nonzero(context_mask).size(0)
    context_nonzero_idx = torch.nonzero(context_mask)
    context_x = context_nonzero_idx.view(num_context, 2).float() / h
    context_y = image_batch[context_mask_cor].view(c, num_context_points)

    target_mask_cor = target_mask.unsqueeze(0).repeat( c, 1, 1)  # repeat for each channel
    num_target = torch.nonzero(target_mask).size(0)
    target_nonzero_idx = torch.nonzero(target_mask)
    target_x = target_nonzero_idx.view( num_target, 2).float() / h
    target_y = image_batch[target_mask_cor].view( c, num_target_points+num_context_points)

    # context_y = context_y.permute(0, 2, 1)#q*255
    # TODO: transpose
    context_y = context_y.transpose(1, 0)#q*255
    target_y = target_y.transpose(1,0)
    # print("context mask: ", context_mask)

    # return ((context_x, context_y), context_x_wo_mask), context_y_wo_mask, context_mask
    context_x = context_x.unsqueeze(0)
    context_y = context_y.unsqueeze(0)
    target_x = target_x.unsqueeze(0)
    target_y = target_y.unsqueeze(0)
    context_mask = context_mask.unsqueeze(0)
    return ((context_x, context_y), target_x), target_y, context_mask, (context_x_wo_mask,context_y_wo_mask)
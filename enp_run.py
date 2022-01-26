import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import time

# All the arguments here
from utilFiles.get_args import the_args
args = the_args()

# Use CUDA if available, otherwise error
assert torch.cuda.is_available()
device = torch.device(f"cuda:{int(args.gpu_id)}" if torch.cuda.is_available() else "cpu")


# Helper functions to save the results, model and load model
from utilFiles.save_load_files_models import save_to_txt_file, save_to_csv_file, save_to_json
from utilFiles.save_load_files_models import save_model, load_model

# All the arguments here
from utilFiles.get_args import the_args
args = the_args()



#The details for the model, the dataset (shared among all the scripts)
from models.shared_model_detail import *

#NP outputs the 4 NIG parameters: (But setting here may not make much difference)
decoder_sizes += [4*args.channels]
print("Decoder sizes: ", decoder_sizes)


from models.np_complete_models import Evd_det_model

model = Evd_det_model(latent_encoder_sizes,
                 determministic_encoder_sizes,
                 decoder_sizes,
                 args,
                 attention,
                 ).to(device)

optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)

# model = load_model("/home/deep/Desktop/IMPLEMENTATION/MAY/ANPHeterogenous/May18/saved_models/model_9000.pth")
if args.load_model:
    model = load_model("name_of_model.pth")

from utilFiles.helper_functions_shared import create_dirs
create_dirs(save_to_dir)

from utilFiles.helper_functions_shared import count_parameters
print("NUm parameters: ", count_parameters(model))


#Save Details
details_txt = str(model) + "\n" + "Num parameters: "+ str(count_parameters(model)) + "\n" + str(args)
save_to_txt_file(f"{save_to_dir}/model_details.txt", details_txt, 0, "Saved Details")

from utilFiles.helper_functions_shared import save_results

if task == "image_completion":
    from plot_functions.plot_2d_image_completion_aug8 import plot_functions_3d_edl
    #Make context set and target set from the image
    from data.task_generator_helpers import get_context_target_2d
elif task == "1d_regression":
    from utilFiles.util_plot_all import plot_functions_alea_ep_1d
    training_iterations = int(args.training_iterations)
    save_models_every = training_iterations//10
else:
    print("Unknown problem")
    raise NotImplementedError

headers = ['Epoch', 'Test Loss', 'Log Likelihood','Epistemic','Aleatoric' ]
logging_dict = {}
for k in headers:
    logging_dict[k] = []

logging_dict_all = []

logging_dict, logging_dict_all = {}, []
def test_model_and_save_results(epoch, tr_time_taken = 0):
    total_test_loss = 0
    total_log_likelihood = 0
    num_test_tasks = 0

    av_epis = 0
    av_alea = 0

    test_time_start = time.time()

    with torch.no_grad():
        model.eval()

        if task == "1d_regression":
            looping_variable = range(args.num_test_tasks)
        elif task == "image_completion":
            looping_variable = enumerate(dataset_test)

        for loop_item in looping_variable:
            model.eval()
            model.zero_grad()
            optimizer.zero_grad()

            if task == "1d_regression":
                index = loop_item
                data_test = dataset_test.generate_curves(device=device, fixed_num_context=args.max_context_points)
                query, target_y = data_test.query, data_test.target_y



            elif task == "image_completion":
                index, (batch_x, batch_label) = loop_item
                batch_x = batch_x.to(device)
                query, target_y, context_mask, _ = get_context_target_2d(batch_x, num_ctx_pts=args.max_context_points)

            # dist, mu, sigma, log_p, kl, loss \
            dist, recons_loss, kl_loss, loss, mu, v, alpha, beta= model(query, None)
            total_log_likelihood += torch.mean(dist.log_prob(target_y))

            test_loss = F.mse_loss(target_y, mu)
            total_test_loss += test_loss
            num_test_tasks += 1

            av_epis += torch.mean(beta / ( v * (alpha - 1)) )
            av_alea += torch.mean(beta / (alpha - 1) )



    average_test_loss = total_test_loss / num_test_tasks
    average_log_likelihood = total_log_likelihood / num_test_tasks
    av_epis /= num_test_tasks
    av_alea /= num_test_tasks

    print("Epoch: {}, test_loss: {}".format(epoch, average_test_loss.detach().cpu().numpy().item()))

    test_time_taken = time.time() - test_time_start

    keys = ["Epoch", "Test Loss", "Test Log Likelihood", "Epistemic", "Aleatoric", "Train Time", "Test Time"]
    values = [epoch]
    values += [float(x.cpu().numpy()) for x in [average_test_loss, average_log_likelihood]]
    values += [float(x.cpu().numpy()) for x in [av_epis, av_alea]]
    values += [tr_time_taken, test_time_taken]

    print("keys: ", keys)
    print("values: ", values)

    global logging_dict
    global logging_dict_all
    if epoch == 0:
        logging_dict = {}
        for k in keys:
            logging_dict[k] = []
        logging_dict_all = []

    logging_dict, logging_dict_all = save_results(logging_dict, logging_dict_all, keys, values, save_to_dir)

    # Now Plotting
    if task == "1d_regression":
        (context_x, context_y), target_x = query
        epis = beta / (v * (alpha - 1))
        alea = beta / (alpha - 1)
        plot_functions_alea_ep_1d(
            target_x.detach().cpu().numpy(),
            data_test.target_y.detach().cpu().numpy(),
            context_x.detach().cpu().numpy(),
            context_y.detach().cpu().numpy(),
            mu.detach().cpu().numpy(),
            epis.detach().cpu().numpy(),
            alea.detach().cpu().numpy(),
            save_img=True,
            save_to_dir=f"{save_to_dir}/saved_images",
            save_name=str(epoch),
        )
    elif task == "image_completion":
        image_one_temp = batch_x
        ch, wdth, ht = batch_x[0].shape
        plot_functions_3d_edl(image_one_temp, mu, v, alpha, beta, target_y, context_mask, epoch,
                              location=f"{save_to_dir}/saved_images/", save=True,
                              w=wdth, h=ht, c=ch)

    return average_test_loss

def one_iteration_training(query, target_y):
    model.train()
    model.zero_grad()
    optimizer.zero_grad()

    _, recons_loss, kl_loss, loss, mu, v, alpha, beta = model(query, target_y, 0)

    loss.backward()
    optimizer.step()

def train_1d_regression(tr_time_end = 0, tr_time_start=0):
    for tr_index in range(args.training_iterations+1):
        save_tracker_val = tr_index % args.test_1d_every
        if save_tracker_val == 0 or tr_index == args.training_iterations:
            tr_time_taken = tr_time_end - tr_time_start
            average_test_loss = test_model_and_save_results(tr_index, tr_time_taken)

            save_model(f"{save_to_dir}/saved_models/model_{tr_index}.pth", model)
            tr_time_start = time.time()
        # Training phase
        data_train = dataset_train.generate_curves(device=device, fixed_num_context=args.max_context_points)
        query, target_y = data_train.query, data_train.target_y

        if args.outlier_training_tasks:
            bs, y_len, dim_3 = target_y.shape

            y_dim = torch.argmax(torch.rand(bs, y_len), dim=1).numpy()

            for i in range(bs):
                target_y[i, y_dim[i], 0] += args.outlier_val  # noise_val

        one_iteration_training(query, target_y)


        tr_time_end = time.time()


def train_image_completion(tr_time_end = 0, tr_time_start=0):
    for epoch in range(args.epochs):

        #Test the model
        if epoch % args.save_results_every == 0 or epoch == args.epochs-1:
            tr_time_taken = tr_time_end - tr_time_start
            average_test_loss = test_model_and_save_results(epoch, tr_time_taken)

        #Save the model
        if epoch % args.save_models_every == 0 or epoch == args.epochs-1:
            save_model(f"{save_to_dir}/saved_models/model_{epoch}.pth", model)

        tr_time_start = time.time()
        # Training phase
        model.train()
        for image_index, (batch_x_instance, _) in enumerate(dataset_train):
            model.zero_grad()
            optimizer.zero_grad()

            batch_x = batch_x_instance.to(device)

            query, target_y,_,_ = get_context_target_2d(batch_x, num_ctx_pts=args.max_context_points)
            # print("target y: ", target_y.shape)

            if args.outlier_training_tasks:
                bs, y_len, dim_3 = target_y.shape

                y_dim = torch.argmax(torch.rand(bs, y_len), dim=1).numpy()

                for i in range(bs):
                    target_y[i, y_dim[i], :] += args.outlier_val  # noise_val

            one_iteration_training(query, target_y)

        tr_time_end = time.time()

def main():
    print("Start Training")
    if task == "1d_regression":
        print("Regression, Dataset: ", args.dataset)
        train_1d_regression()

    elif task == "image_completion":
        print("Image Completion, Dataset: ", args.dataset)
        train_image_completion()
    pass


if __name__ == "__main__":
    main()

import torch
import torch.optim as optim
import torch.nn.functional as F
import time
import numpy as np

# All the arguments here
from utilFiles.get_args import the_args
args = the_args()

# Use CUDA if available, otherwise error
assert torch.cuda.is_available()
device = torch.device(f"cuda:{int(args.gpu_id)}" if torch.cuda.is_available() else "cpu")

# Helper functions to save the results, model and load model
from utilFiles.save_load_files_models import save_to_txt_file
from utilFiles.save_load_files_models import save_model, load_model


# If deterministic run needed
from utilFiles.set_deterministic import make_deterministic
make_deterministic(args.seed)


#The details for the model, the dataset (shared among all the scripts)
from models.shared_model_detail import *

#NP outputs the 2 Gaussian parameters: mean and variance
decoder_sizes += [2*args.channels]
print("Decoder sizes: ", decoder_sizes)

from models.np_complete_models import Evd_lat_model

model = Evd_lat_model(latent_encoder_sizes,
                 determministic_encoder_sizes,
                 decoder_sizes,
                 args,
                 attention,
                 ).to(device)

optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)

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
    from plot_functions.plot_2d_image_completion_aug8 import plot_functions_3d
    #Make context set and target set from the image
    from data.task_generator_helpers import get_context_target_2d
elif task == "1d_regression":
    from utilFiles.util_plot_all import plot_functions_alea_ep_1d
    training_iterations = int(args.training_iterations)
    save_models_every = training_iterations//10
else:
    print("Unknown problem")
    raise NotImplementedError


logging_dict, logging_dict_all = {}, []
def test_model_and_save_results(epoch, tr_time_taken = 0, train_stats = {}):
    total_test_loss = 0
    total_log_likelihood = 0
    num_test_tasks = 0

    av_epis =0
    av_alea =0

    test_time_start = time.time()

    if args.use_latent_path:
        ctx_epis_lat_avg, ctx_alea_lat_avg, tar_epis_lat_avg, tar_alea_lat_avg = 0,0,0,0

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

            if args.use_latent_path:
                mu = 0
                lat_log_likelihood = 0
                means = []
                variances = []
                num_mc_test_samples = 5
                for _ in range(num_mc_test_samples):
                    dist, m, sigma, latent_uncertainties, tr_keys = model(query, None)
                    lat_log_likelihood += torch.mean(dist.log_prob(target_y))
                    mu += m
                    means.append(m)
                    variances.append(sigma)
                mu /= num_mc_test_samples
                lat_log_likelihood /= num_mc_test_samples
                total_log_likelihood += lat_log_likelihood
                ((ctx_epis_lat, ctx_alea_lat), (tar_epis_lat, tar_alea_lat)) = latent_uncertainties

                #Latent Stats
                ctx_epis_lat_avg += ctx_epis_lat
                ctx_alea_lat_avg += ctx_alea_lat
                tar_epis_lat_avg += tar_epis_lat
                tar_alea_lat_avg += tar_alea_lat

            else:
                print("Latent Evidential Model Needs Latent Path!")
                raise NotImplementedError

            stack_means = torch.stack(means)
            # print("means: ", means, means.shape)
            var_means = torch.var(stack_means,dim=0)
            # print("var means: ", var_means, var_means.shape)
            # print("Means: ", torch.var(means, dim = 0).shape, m.shape)
            av_epis += torch.mean(var_means)
            av_alea += torch.mean(torch.stack(variances))
            test_loss = F.mse_loss(target_y, mu)
            total_test_loss += test_loss
            num_test_tasks += 1

    average_test_loss = total_test_loss / num_test_tasks
    average_log_likelihood = total_log_likelihood / num_test_tasks
    av_epis /= num_test_tasks
    av_alea /= num_test_tasks



    print("Epoch: {}, test_loss: {}".format(epoch, average_test_loss.detach().cpu().numpy().item()))

    test_time_taken = time.time() - test_time_start

    keys = ["Epoch", "Test Loss", "Test Log Likelihood", "Epistemic", "Aleatoric", "Train Time", "Test Time"]
    values = [epoch]
    values += [float(x.cpu().numpy()) for x in [average_test_loss, average_log_likelihood]]
    values += [float(x.cpu().numpy()) for x in  [av_epis, av_alea]]
    values += [tr_time_taken, test_time_taken]

    if args.use_latent_path:
        ctx_epis_lat_avg /= num_test_tasks
        ctx_alea_lat_avg /= num_test_tasks

        #Test time Target epistemic is not meaningful (not available as well)
        tar_epis_lat_avg /= num_test_tasks
        tar_alea_lat_avg /= num_test_tasks



        keys += ["Ctx Lat Epis", "Ctx Lat Alea"]
        values += [x for x in [ctx_epis_lat_avg, ctx_alea_lat_avg]]

    if train_stats != {}:
        for k, v in train_stats.items():
            keys += [k]
            values += [v] #[float(v.detach().cpu().numpy())]
    else:
        for k in tr_keys:
            keys += [k]
            values += [0]


    global logging_dict
    global logging_dict_all
    if epoch == 0:
        logging_dict = {}
        for k in keys:
            logging_dict[k] = []
        logging_dict_all = []

    print("keys: ", keys)
    print("values: ", values)

    logging_dict, logging_dict_all = save_results(logging_dict, logging_dict_all, keys, values, save_to_dir)

    # Save Images
    if task == "1d_regression":
        (context_x, context_y), target_x = data_test.query
        epis = torch.var(torch.stack(means), dim=0)
        alea = torch.mean(torch.stack(variances), dim=0)
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
        # Save Images
        image_one_temp = batch_x
        ch, wdth, ht = image_one_temp[0].shape
        plot_functions_3d(image_one_temp, mu, sigma, context_mask, epoch, location=f"{save_to_dir}/saved_images/",
                          save=True,
                          w=wdth, h=ht, c=ch)

    return average_test_loss

def one_iteration_training(query, target_y):
    model.train()
    model.zero_grad()
    optimizer.zero_grad()

    loss_total = 0
    if args.use_latent_path:
        num_mc_samples_train = 5

        loss_dict_avg = {}

        for _ in range(num_mc_samples_train):
            dist, loss, mu, sigma, latent_uncertainties, loss_dict = model(query, target_y)
            if loss_dict_avg == {}:
                loss_dict_avg = loss_dict
            else:
                for k, v in loss_dict.items():
                    loss_dict_avg[k] += v
            loss_total += loss
        loss_total /= num_mc_samples_train
        for k,v in loss_dict_avg.items():
            loss_dict_avg[k] /= num_mc_samples_train
    else:
        print("Latent Evidential Model Needs Latent Path")
        raise NotImplementedError

    loss_total.backward()
    optimizer.step()
    return loss_dict_avg

def train_1d_regression(tr_time_end = 0, tr_time_start=0):
    count_iter = 0
    av_loss_stats = {}
    for tr_index in range(args.training_iterations+1):
        save_tracker_val = tr_index % args.test_1d_every
        if (save_tracker_val == 0 or tr_index == args.training_iterations):
            tr_time_taken = tr_time_end - tr_time_start
            train_stats = {}
            if av_loss_stats != {}:
                for k, v in av_loss_stats.items():
                    train_stats[k] = v / count_iter
            count_iter = 0
            av_loss_stats = {}

            _ = test_model_and_save_results(tr_index, tr_time_taken, train_stats)

            save_model(f"{save_to_dir}/saved_models/model_{tr_index}.pth", model)
            tr_time_start = time.time()
        # Training phase
        data_train = dataset_train.generate_curves(device=device, fixed_num_context=args.max_context_points)
        query, target_y = data_train.query, data_train.target_y

        if args.outlier_training_tasks:
            bs, y_len, dim_3 = target_y.shape

            y_dim = torch.argmax(torch.rand(bs, y_len), dim = 1).numpy()

            for i in range(bs):
                target_y[i, y_dim[i], 0] += args.outlier_val  # noise_val

        one_it_loss_stat = one_iteration_training(query, target_y)

        count_iter+= 1
        if av_loss_stats == {}:
            av_loss_stats = one_it_loss_stat
        else:
            for k, v in one_it_loss_stat.items():
                av_loss_stats[k] += v

        tr_time_end = time.time()

def train_image_completion(tr_time_end = 0, tr_time_start=0):
    av_loss_stats = {}
    for epoch in range(args.epochs):

        #Test the model
        if( epoch % args.save_results_every == 0 or epoch == args.epochs-1):
            tr_time_taken = tr_time_end - tr_time_start

            if av_loss_stats == {}:
                train_stats = {}
            else:
                for k, v in av_loss_stats.items():
                    train_stats[k] = v / count_iter
            count_iter = 0
            av_loss_stats = {}

            _ = test_model_and_save_results(epoch, tr_time_taken, train_stats)

        #Save the model
        if epoch % args.save_models_every == 0 or epoch == args.epochs-1:
            save_model(f"{save_to_dir}/saved_models/model_{epoch}.pth", model)

        tr_time_start = time.time()
        count_iter = 0
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

            one_it_loss_stat = one_iteration_training(query, target_y)

            count_iter += 1
            if av_loss_stats == {}:
                av_loss_stats = one_it_loss_stat
            else:
                for k, v in one_it_loss_stat.items():
                    av_loss_stats[k] += v
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

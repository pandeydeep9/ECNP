def the_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', '--dataset', type=str, default='mnist', help="1d-sin, gp, mnist, cifar10, celeba..The dataset name")
    parser.add_argument('-seed', '--seed', type=int, default=0, help='Seed for experiment')
    parser.add_argument('-tr_it', '--training_iterations', type=int, default=20000, help='Number of training iterations')
    parser.add_argument('-num_epochs', '--num_epochs', type=int, default=50, help='Number of training iterations')

    parser.add_argument('-test_1d_every', '--test_1d_every', type=int, default=2000,
                        help='How often to save  test, logs, and save figures (iteration for 1d)')
    parser.add_argument('-save_results_every', '--save_results_every', type=int, default=1,
                        help='How often to save logs/figures (epoch for 2d, iteration for 1d)')
    parser.add_argument('-num_test_tasks', '--num_test_tasks', type=int, default=2000,
                        help='Number of test tasks to average on (only 1d)')

    parser.add_argument('-max_context_points', '--max_context_points', type=int, default=50,
                        help='Maximum number of context points')

    parser.add_argument('-model_type', '--model_type', type=str, default='CNP', help='Model type: ANP, NP, CNP')
    parser.add_argument('-attention_type', '--attention_type', type=str, default='multihead',
                        help='Multhihead, uniform,dot,laplace')

    parser.add_argument('-gpu','--gpu_id', type=int,default=0, help="The id for the gpu")

    parser.add_argument('-e', '--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('-b', '--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('-name', '--experiment_name', type=str, default='CNP-model-save-name', help='name_for_experiment')



    # MODEL details
    parser.add_argument('-rps', '--representation_size', type=int, default=128, help='Representation size for context')
    parser.add_argument('-hs', '--hidden_size', type=int, default=128, help='Model hidden size')
    parser.add_argument('-nmhdnlrs', '--num_enc_hdn_lrs', type=int, default=4,
                        help='Number of hidden layers for encoder')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001, help='Learning Rate for experiment')

    parser.add_argument('-use_det', '--use_deterministic_path', type=str, default='True', help='Use deterministic path')
    parser.add_argument('-use_lat', '--use_latent_path', type=str, default='False', help='Use latent path')



    parser.add_argument('-debug', '--debugging', type=str, default="False", help="whether to debug each iteration")
    parser.add_argument('-load_mdl', '--load_model', type=str, default="False",
                        help="whether to load a model or scratch train")

    parser.add_argument('-nig_nll_reg_coef', '--nig_nll_reg_coef', type=float, default=0.1,
                        help="EDL nll reg balancing factor")
    parser.add_argument('-nig_nll_ker_reg_coef', '--nig_nll_ker_reg_coef', type=float, default=1.0,
                        help='EDL kernel reg balancing factor')

    # parser.add_argument('-beta_constraint_add', '--beta_constraint_add', type=float, default=0.1, help="EDL output beta constraint")
    parser.add_argument('-ev_dec_beta_min', '--ev_dec_beta_min', type=float, default=0.2,
                        help="EDL Decoder beta minimum value")
    parser.add_argument('-ev_dec_alpha_max', '--ev_dec_alpha_max', type=float, default=20.0,
                        help="EDL output alpha maximum value")
    parser.add_argument('-ev_dec_v_max', '--ev_dec_v_max', type=float, default=20.0, help="EDL output v maximum value")

    parser.add_argument('-ev_lat_beta_min', '--ev_lat_beta_min', type=float, default=0.2,
                        help="EDL Latent beta minimum value")
    parser.add_argument('-ev_lat_alpha_max', '--ev_lat_alpha_max', type=float, default=20.0,
                        help="EDL Latent alpha maximum value")
    parser.add_argument('-ev_lat_v_max', '--ev_lat_v_max', type=float, default=20.0, help="EDL output v maximum value")

    #noisy_training_tasks
    parser.add_argument('-outlier_training_tasks', '--outlier_training_tasks', default="False", type=str, help="Are there outliers in training tasks(True/False)")
    parser.add_argument('-outlier_val', '--outlier_val', default=0.0, type=float, help="How extreme are the outliers (add)")
    parser.add_argument('-active_task_sel', '--active_task_sel', type=str, default='False', help='Do Task Selection?')
    parser.add_argument('-use_domain_knowledge', '--use_domain_knowledge', type=str, default='False', help = 'Use specific tasks')

    args = parser.parse_args()
    args.use_deterministic_path = args.use_deterministic_path.lower() == 'true'
    args.use_latent_path = args.use_latent_path.lower() == 'true'

    args.debugging = args.debugging.lower() == 'true'

    args.load_model = args.load_model.lower() == 'true'
   

    args.active_task_sel = args.active_task_sel.lower() == "true"
    args.use_domain_knowledge = args.use_domain_knowledge.lower() == "true"


    ###Later add
    args.training_iterations = int(args.training_iterations)
    args.save_models_every = 5#%(args.num_epochs - 1) // 5
    # args.save_models_every = 1

    return args

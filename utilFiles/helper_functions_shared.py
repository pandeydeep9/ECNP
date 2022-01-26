import os
def create_dirs(save_to_dir):
    if not os.path.exists(f"{save_to_dir}"):
        print("Create new")
        os.makedirs(f"{save_to_dir}")
        os.makedirs(f"{save_to_dir}/saved_models")
        os.makedirs(f"{save_to_dir}/saved_images")
    else:
        print("Save To existing")


def count_parameters(model):
    print(model)
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

from utilFiles.save_load_files_models import save_to_txt_file, save_to_csv_file, save_to_json
def save_results(logging_dict, logging_dict_all, keys, values, save_to_dir):
    logging_dict_one = {}
    for k, v in zip(keys, values):
        logging_dict[k] = logging_dict[k] + [v]
        logging_dict_one[k] = v

    logging_dict_all.append(logging_dict_one)

    save_to_csv_file(f"{save_to_dir}/results.csv", logging_dict_all)
    save_to_json(f"{save_to_dir}/save_json.json", logging_dict)

    return logging_dict, logging_dict_all
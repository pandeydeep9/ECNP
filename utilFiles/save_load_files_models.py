import csv
import torch
import numpy as np
import os

import json

def save_to_txt_file(filename, value, iteration=0, header="None"):
    file = open(filename, 'a')
    if iteration == 0:
        file.write(str(header)+"\n")
    file.write(str(value)+"\n")
    # file.writelines("\n")
    file.close()

def save_to_csv_file(filename, logging_dict):
    key = logging_dict[0].keys()
    with open(filename, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=key)
        writer.writeheader()
        for a in logging_dict:
            writer.writerow(a)



def save_to_json(filename, dict_to_store):
    with open(os.path.abspath(filename), 'w') as f:
        json.dump(dict_to_store, fp=f)


# Save logs to file (For debugging during train, results during test)
def save_dict_to_csv_file(logging_dict, experiment_name, iteration=-1, create = False):
    line_to_add = logging_dict.keys()
    value_to_add = logging_dict.values()

    file_to_save = experiment_name + "dict_to_csv_save.csv"
    # print("saving at: ", file_to_save)
    file_exists = os.path.isfile(file_to_save)
    if iteration == 0 or create or not file_exists:
        with open(file_to_save, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(line_to_add)
            writer.writerow(value_to_add)
    else:
        with open(file_to_save, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(value_to_add)


def save_model( path, model):
    torch.save({'state_dict': model.state_dict()}, path+'.pth.tar')
    torch.save(model, path)

def load_model(path):
    model=torch.load(path)
    return model



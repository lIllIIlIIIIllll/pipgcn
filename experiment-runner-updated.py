import os
import sys
import yaml
import traceback
import cPickle
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from configuration import data_directory, experiment_directory, output_directory, seeds, printt
from train_test import TrainTest
from pw_classifier import PWClassifier
from results_processor import ResultsProcessor

def get_available_gpus():
    """Get list of available GPU devices."""
    from tensorflow.python.client import device_lib
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

# Load experiment specified in system args
exp_file = sys.argv[1]
printt("Running Experiment File: {}".format(exp_file))
f_name = exp_file.split(".")[0] if "." in exp_file else exp_file
exp_specs = yaml.load(open(os.path.join(experiment_directory, exp_file), 'r').read())

# setup output directory
outdir = os.path.join(output_directory, f_name)
if not os.path.exists(outdir):
    os.mkdir(outdir)
results_processor = ResultsProcessor()

# create results log
results_log = os.path.join(outdir, "results.csv")
with open(results_log, 'w') as f:
    f.write("")

# write experiment specifications to file
with open(os.path.join(outdir, "experiment.yml"), 'w') as f:
    f.write("{}\n".format(yaml.dump(exp_specs)))

# Get available GPUs
available_gpus = get_available_gpus()
num_gpus = len(available_gpus)
printt(f"Found {num_gpus} GPUs: {available_gpus}")

# Calculate total experiments for progress bar
total_experiments = len(exp_specs["experiments"]) * len(seeds)
pbar = tqdm(total=total_experiments, desc="Experiments Progress")

# perform each experiment
prev_train_data_file = ""
prev_test_data_file = ""
first_experiment = True

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = ','.join([str(i) for i in range(num_gpus)])

for name, experiment in exp_specs["experiments"]:
    train_data_file = os.path.join(data_directory, experiment["train_data_file"])
    test_data_file = os.path.join(data_directory, experiment["test_data_file"])
    try:
        # Reuse train data if possible
        if train_data_file != prev_train_data_file:
            printt("Loading train data")
            train_list, train_data = cPickle.load(open(train_data_file))
            prev_train_data_file = train_data_file
        if test_data_file != prev_test_data_file:
            printt("Loading test data")
            test_list, test_data = cPickle.load(open(test_data_file))
            prev_test_data_file = test_data_file
            
        # create data dictionary
        data = {"train": train_data, "test": test_data}
        
        # perform experiment for each random seed
        for i, seed_pair in enumerate(seeds):
            printt("{}: rep{}".format(name, i))
            
            # Distribute across GPUs using device scope
            gpu_idx = i % num_gpus
            with tf.device(f'/gpu:{gpu_idx}'):
                # set tensorflow and numpy seeds
                tf.set_random_seed(seed_pair["tf_seed"])
                np.random.seed(int(seed_pair["np_seed"]))
                
                printt(f"Building model on GPU {gpu_idx}")
                # build model with specified config
                with tf.Session(config=config) as sess:
                    model = PWClassifier(experiment["layers"], experiment["layer_args"], 
                                       data["train"], 0.1, 0.1, outdir)
                    # train and test the model
                    headers, results = TrainTest(results_processor).fit_model(exp_specs, data, model)
                    
                    # write headers to file if haven't already
                    if first_experiment:
                        with open(results_log, 'a') as f:
                            f.write("{}\n".format(",".join(["file", "experiment", "rep", "specifications"] + headers)))
                        first_experiment = False
                        
                    # write results to file
                    with open(results_log, 'a') as f:
                        f.write("{}, {}, {}, {}, {}\n".format(
                            f_name, name, i, 
                            format(experiment).replace(",", ";"), 
                            ",".join([str(r) for r in results])
                        ))
            
            # Update progress bar
            pbar.update(1)
            
    except Exception as er:
        if er is KeyboardInterrupt:
            raise er
        ex_str = traceback.format_exc()
        printt(ex_str)
        printt("Experiment failed: {}".format(exp_specs))

pbar.close()

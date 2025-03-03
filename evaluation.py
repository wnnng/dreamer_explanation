import copy
import pathlib
import pickle
import warnings

import elements
import numpy as np
from natsort import natsorted

from dreamerv3.explainer import collect_run_data, generate_explanation, \
    load_agent, update_config, make_env, make_agent, load_checkpoint

warnings.simplefilter('ignore')
folder = pathlib.Path.cwd()
np.random.seed(598234390)
def get_subdirectories(path):
    return [f.name for f in pathlib.Path(path).iterdir() if f.is_dir()]


benchmarks_list = ['minigrid_DoorKey-16x16_2', 'minigrid_SimpleCrossingS9N1', 'minigrid_LavaCrossingS9N1',
                   'minigrid_SimpleCrossingS11N5', 'minigrid_DoorKey-5x5', ]
path = 'evaluation/checkpoints/'
subdirs = get_subdirectories(path)
logdir = []
checkpoints = []
eval_checkpoints = []
for s in subdirs:
    if s in benchmarks_list:
        ld = path + s
        logdir.append(ld)
        c = get_subdirectories(ld + "/ckpt/")
        if len(c) > 1:
            raise ValueError
        checkpoints.append(ld + "/ckpt/" + c[0])
        ec = get_subdirectories(ld + "/eval_ckpt/")
        ec_list = []
        for e in ec:
            ec_list.append(ld + "/eval_ckpt/" + e + "/cp")
        eval_checkpoints.append(ec_list)
benchmarks = {}
for i, b in enumerate(benchmarks_list):
    benchmarks[b] = [logdir[i], checkpoints[i], eval_checkpoints[i]]

# Correctness - Model Replacement Check
#
# We use a different environment during the observation, the explanation should show that it is different and provide
experiments = []
carry_list = []
data_list = []
recorded_obs_list = []
recorded_act_list = []
recorded_outs_list = []
benchmarks_names = []
for k, v in benchmarks.items():
    for i in range(25):
        benchmarks_names.append(k)

    config = update_config(folder, k.split('_')[0], v[0], k)
    agent = make_agent(config)
    env = make_env(config, 0)
    act_space = env.act_space
    cp = elements.Checkpoint()
    cp.agent = agent
    cp.load(v[1], keys=['agent'])
    for i in range(25):
        carry, data, recorded_obs, recorded_act, recorded_outs = collect_run_data(agent, env, act_space)
        carry_list.append(carry)
        data_list.append(data)
        recorded_obs_list.append(recorded_obs)
        recorded_act_list.append(recorded_act)
        recorded_outs_list.append(recorded_outs)

for k2, v2 in benchmarks.items():
    config = update_config(folder, k2.split('_')[0], v2[0], k2)
    agent = make_agent(config)
    cp.agent = agent
    cp.load(v2[1], keys=['agent'])
    for i in range(len(data_list)):
        carry, explanation_combined, action = generate_explanation(agent, carry_list[i], data_list[i], 15)
        experiments.append(
            {'benchmark': benchmarks_names[i], 'benchmark_eval': k2, 'run': i, 'recorded_obs': recorded_obs_list[i],
             'recorded_act': recorded_act_list[i],
             'recorded_outs': recorded_outs_list[i], 'explanations': explanation_combined})
with open('evaluation/MRC.plk', 'wb') as f:
    pickle.dump(experiments, f)

# # Correctness - Model Randomization Check
# We replace the parts of the models with an untrained checkpoint to see, how the explanation changes.
experiments = []
for k, v in benchmarks.items():
    carry_list = []
    data_list = []
    recorded_obs_list = []
    recorded_act_list = []
    recorded_outs_list = []
    agent, env, act_space = load_agent(folder, v[0], k.split('_')[0], k, v[1])
    for i in range(25):
        carry, data, recorded_obs, recorded_act, recorded_outs = collect_run_data(agent, env, act_space)
        carry_list.append(carry)
        data_list.append(data)
        recorded_obs_list.append(recorded_obs)
        recorded_act_list.append(recorded_act)
        recorded_outs_list.append(recorded_outs)
        expl_carry, explanation_combined, action = generate_explanation(agent, carry, data, 15)
        experiments.append(
            {'benchmark': k, 'run': 0, 'iter': i, 'recorded_obs': recorded_obs, 'recorded_act': recorded_act,
             'recorded_outs': recorded_outs, 'explanations': explanation_combined})
    #Replace Actor
    agent, env, act_space = load_agent(folder, v[0], k.split('_')[0], k, v[1])
    buffer = pathlib.Path(v[2][0] + "/cp/agent.pkl").read_bytes()
    agent_data = pickle.loads(buffer)
    random = copy.copy(agent_data)
    for key in random['params']:
        random['params'][key] = np.random.rand(*random['params'][key].shape)
    agent.load(random, regex=r"^pol($|/)")
    for i in range(len(data_list)):
        expl_carry, explanation_combined, action = generate_explanation(agent, carry_list[i], data_list[i], 15)
        experiments.append(
            {'benchmark': k, 'run': 1, 'iter': i, 'recorded_obs': recorded_obs, 'recorded_act': recorded_act,
             'recorded_outs': recorded_outs, 'explanations': explanation_combined})
    #Replace World Model
    agent, env, act_space = load_agent(folder, v[0], k.split('_')[0], k, v[1])
    agent.load(random, regex=r"^dyn($|/)")
    for i in range(len(data_list)):
        expl_carry, explanation_combined, action = generate_explanation(agent, carry_list[i], data_list[i], 15)
        experiments.append(
            {'benchmark': k, 'run': 2, 'iter': i, 'recorded_obs': recorded_obs, 'recorded_act': recorded_act,
             'recorded_outs': recorded_outs, 'explanations': explanation_combined})
    #Replace Reward Predictor
    agent, env, act_space = load_agent(folder, v[0], k.split('_')[0], k, v[1])
    agent.load(random, regex=r"^rew($|/)")
    for i in range(len(data_list)):
        expl_carry, explanation_combined, action = generate_explanation(agent, carry_list[i], data_list[i], 15)
        experiments.append(
            {'benchmark': k, 'run': 3, 'iter': i, 'recorded_obs': recorded_obs, 'recorded_act': recorded_act,
             'recorded_outs': recorded_outs, 'explanations': explanation_combined})
    #Replace Value
    agent, env, act_space = load_agent(folder, v[0], k.split('_')[0], k, v[1])
    agent.load(random, regex=r"^val($|/)")
    for i in range(len(data_list)):
        expl_carry, explanation_combined, action = generate_explanation(agent, carry_list[i], data_list[i], 15)
        experiments.append(
            {'benchmark': k, 'run': 4, 'iter': i, 'recorded_obs': recorded_obs, 'recorded_act': recorded_act,
             'recorded_outs': recorded_outs, 'explanations': explanation_combined})
    #Replace continue
    agent, env, act_space = load_agent(folder, v[0], k.split('_')[0], k, v[1])
    agent.load(random, regex=r"^con($|/)")
    for i in range(len(data_list)):
        expl_carry, explanation_combined, action = generate_explanation(agent, carry_list[i], data_list[i], 15)
        experiments.append(
            {'benchmark': k, 'run': 5, 'iter': i, 'recorded_obs': recorded_obs, 'recorded_act': recorded_act,
             'recorded_outs': recorded_outs, 'explanations': explanation_combined})
with open('evaluation/MPRC.plk', 'wb') as f:
    pickle.dump(experiments, f)

# # Consistency - Input Perturbation
#
# We record two trajectories, one with the normal images and one with noisy images.
# We create explanations for both. The explanations should be similar.
experiments = []
perturbations = [0.1, 0.2, 0.3, 0.4]
for k, v in benchmarks.items():

    config = update_config(folder, k.split('_')[0], v[0], k, {'env.minigrid.use_seed': False})
    agent = make_agent(config)
    load_checkpoint(agent, v[1])
    for i in range(25):
        seed = np.random.randint(100000)
        for per in perturbations:
            env = make_env(config, 0, seed=seed)
            act_space = env.act_space
            carry, data, recorded_obs, recorded_act, recorded_outs = collect_run_data(agent, env, act_space,
                                                                                      obs_perturbation=per)
            expl_carry, explanation_combined, action = generate_explanation(agent, carry, data, 15)
            experiments.append(
                {'benchmark': k, 'run': i, 'perturbation': per, 'seed': seed, 'recorded_obs': recorded_obs,
                 'recorded_act': recorded_act,
                 'recorded_outs': recorded_outs, 'explanations': explanation_combined})
with open(f'evaluation/IPC.plk', 'wb') as f:
    pickle.dump(experiments, f)


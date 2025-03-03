import importlib
import pathlib
import warnings

import elements
import lpips
import numpy as np
import tables as tb
import torch
import torchvision.transforms as transforms
from ruamel import yaml
from scipy.stats import entropy

from dreamerv3.main import wrap_env

warnings.simplefilter('ignore')
from PIL import Image

loss_fn = lpips.LPIPS(net="alex")

transform = transforms.Compose(
    [
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 2 - 1),
    ]
)


class Step(tb.IsDescription):
    experiment_id = tb.UInt32Col()
    experiment_run_id = tb.Int32Col()
    run_id = tb.Int8Col()
    step_id = tb.Int32Col()
    image = tb.UInt8Col(shape=(64, 64, 3))
    reward = tb.Float32Col()
    is_first = tb.BoolCol()
    is_last = tb.BoolCol()
    is_terminal = tb.BoolCol()
    action = tb.Int8Col()


class MinigridStep(Step):
    direction = tb.UInt8Col()
    action_probs = tb.Float32Col(shape=(7,))
    action_logits = tb.Float32Col(shape=(7,))


class ExplanationStep(tb.IsDescription):
    run_id = tb.Int32Col()
    main_step = tb.Int32Col()
    explanation_idx = tb.Int32Col()
    image = tb.UInt8Col(shape=(64, 64, 3))
    deter = tb.Float32Col(shape=(2048,))
    logit = tb.Float32Col(shape=(32, 16))
    stoch = tb.Int8Col(shape=(32, 16))
    action = tb.Int32Col()
    con = tb.Float32Col()
    value = tb.Float32Col(dflt=np.nan)
    reward = tb.Float32Col(dflt=np.nan)
    image_mse = tb.Float32Col(dflt=np.nan)
    image_lpips = tb.Float32Col(dflt=np.nan)
    deter_mse = tb.Float32Col(dflt=np.nan)
    logit_mse = tb.Float32Col(dflt=np.nan)
    stoch_distance = tb.Float32Col(dflt=np.nan)
    action_logits_mse = tb.Float32Col(dflt=np.nan)
    action_probs_kl = tb.Float32Col(dflt=np.nan)
    action_diff = tb.BoolCol(dflt=True)
    value_diff = tb.Float32Col(dflt=np.nan)
    rew_diff = tb.Float32Col(dflt=np.nan)
    rew_sum = tb.Float32Col(dflt=np.nan)
    rew_sum_diff = tb.Float32Col(dflt=np.nan)


class MinigridExplanationStep(ExplanationStep):
    action_probs = tb.Float32Col(shape=(7,))
    action_logits = tb.Float32Col(shape=(7,))


class HighwayExplanationStep(ExplanationStep):
    deter = tb.Float32Col(shape=(8192,))
    logit = tb.Float32Col(shape=(32, 64))
    stoch = tb.Int8Col(shape=(32, 64))
    action_probs = tb.Float32Col(shape=(3,))
    action_logits = tb.Float32Col(shape=(3,))


class MergeExplanationStep(HighwayExplanationStep):
    action_probs = tb.Float32Col(shape=(5,))
    action_logits = tb.Float32Col(shape=(5,))


class ProcgenExplanationStep(ExplanationStep):
    image = tb.UInt8Col(shape=(96, 96, 3))
    action_probs = tb.Float32Col(shape=(15,))
    action_logits = tb.Float32Col(shape=(15,))


def _mask(value, mask):
    while mask.ndim < value.ndim:
        mask = mask[..., None]
    return value * mask.astype(value.dtype)


def experiments_to_hdf5(file_path, experiments):
    with tb.open_file(file_path, mode="w") as h5file:
        benchmarks = []
        for e_numb, experiment in enumerate(experiments):
            if experiment["benchmark"] not in benchmarks:
                exp_group = h5file.create_group(
                    "/",
                    f"experiment_{len(benchmarks)}",
                    f"Experiment {len(benchmarks)} Data",
                )
                exp_group._v_attrs["benchmark"] = experiment["benchmark"]
                benchmarks.append(experiment["benchmark"])

            run_group = h5file.create_group(
                exp_group, f"run_{experiment['run']}", f"Run {experiment['run']} Data"
            )

            # Create a table for main steps
            if "minigrid" in experiment["benchmark"].lower():
                step_table = h5file.create_table(
                    run_group, "steps", MinigridStep, "Steps Data"
                )
            else:
                step_table = h5file.create_table(run_group, "steps", Step, "Steps Data")
            step_row = step_table.row

            if "minigrid" in experiment["benchmark"].lower():
                explanation_table = h5file.create_table(
                    run_group, "explanations", MinigridExplanationStep, "Explanations Data"
                )
            else:
                explanation_table = h5file.create_table(
                    run_group, "explanations", ExplanationStep, "Explanations Data"
                )

            expl_row = explanation_table.row

            num_steps = len(experiment["recorded_obs"]["image"][0])

            for i in range(num_steps):
                step_row["experiment_id"] = len(benchmarks)
                step_row["experiment_run_id"] = experiment["run"]
                step_row["run_id"] = e_numb
                step_row["step_id"] = i
                step_row["image"] = experiment["recorded_obs"]["image"][0][i]
                if "minigrid" in experiment["benchmark"].lower():
                    step_row["direction"] = experiment["recorded_obs"]["direction"][0][i]
                step_row["reward"] = experiment["recorded_obs"]["reward"][0][i]
                step_row["is_first"] = experiment["recorded_obs"]["is_first"][0][i]
                step_row["is_last"] = experiment["recorded_obs"]["is_last"][0][i]
                step_row["is_terminal"] = experiment["recorded_obs"]["is_terminal"][0][i]
                step_row["action"] = experiment["recorded_act"]["action"][0][i]
                step_row["action_logits"] = experiment['recorded_outs']['logits'][0][i]
                step_row["action_probs"] = experiment['recorded_outs']['probs'][0][i]

                step_row.append()

                if i != num_steps:
                    num_expl = len(experiment["explanations"]["image"][i])
                    for j in range(num_expl):
                        expl_row["run_id"] = e_numb
                        expl_row["main_step"] = i
                        expl_row["explanation_idx"] = j + i
                        expl_row["image"] = experiment["explanations"]["image"][i][j]
                        if j == 0:
                            expl_row["deter"] = experiment['explanations']['obsfeat']['deter'][0][i]
                            expl_row["logit"] = experiment["explanations"]['obsfeat']['logit'][0][i]
                            expl_row["stoch"] = experiment["explanations"]['obsfeat']['stoch'][0][i]
                            expl_row["action"] = experiment["explanations"]["action"][i][j]
                            expl_row["action_logits"] = experiment['explanations']['logits'][i][j]
                            expl_row["action_probs"] = experiment['explanations']['probs'][i][j]
                            expl_row["action_logits_mse"] = np.mean(
                                (experiment['explanations']['logits'][i][j] - experiment['recorded_outs']['logits'][0][
                                    i + j]) ** 2)
                            expl_row["action_probs_kl"] = entropy(experiment['recorded_outs']['probs'][0][
                                                                      i + j],
                                                                  experiment['explanations']['probs'][i][j])
                            expl_row["action_diff"] = not experiment["explanations"]["action"][i][j] == \
                                                          experiment["recorded_act"]["action"][0][i + j]
                            expl_row["con"] = 1.0
                            expl_row["image_mse"] = np.mean((experiment['explanations']['image'][i][j] -
                                                             experiment['recorded_obs']["image"][0][i + j]) ** 2)
                            expl_row["deter_mse"] = 0
                            expl_row["logit_mse"] = 0
                            expl_row["stoch_distance"] = 0
                        elif j == num_expl - 1:
                            expl_row["deter"] = experiment['explanations']['imgfeat']['deter'][i][j - 1]
                            expl_row["logit"] = experiment["explanations"]['imgfeat']['logit'][i][j - 1]
                            expl_row["stoch"] = experiment["explanations"]['imgfeat']['stoch'][i][j - 1]
                            expl_row["con"] = experiment["explanations"]["con"][i][j - 1]
                            expl_row["value"] = experiment["explanations"]["val"][i][j - 1]
                            expl_row["reward"] = experiment["explanations"]["reward"][i][j - 1]
                            try:
                                expl_row["action"] = experiment["explanations"]["action"][i][j]
                                expl_row["action_logits"] = experiment['explanations']['logits'][i][j]
                                expl_row["action_probs"] = experiment['explanations']['probs'][i][j]
                                expl_row["action_logits_mse"] = np.mean((experiment['explanations']['logits'][i][j] -
                                                                         experiment['recorded_outs']['logits'][0][
                                                                             i + j]) ** 2)
                                expl_row["action_probs_kl"] = entropy(
                                    experiment['recorded_outs']['probs'][0][i + j],
                                    experiment['explanations']['probs'][i][j])
                                expl_row["action_diff"] = not experiment["explanations"]["action"][i][j] == \
                                                              experiment["recorded_act"]["action"][0][i + j]
                            except IndexError:
                                expl_row["action"] = -1
                                try:
                                    expl_row["action_diff"] = not experiment["recorded_act"]["action"][0][i + j] == -1
                                except IndexError:
                                    expl_row["action_diff"] = True
                            try:
                                expl_row["image_mse"] = np.mean((experiment['explanations']['image'][i][j] -
                                                                 experiment['recorded_obs']["image"][0][i + j]) ** 2)
                                expl_row["deter_mse"] = np.mean((experiment['explanations']['imgfeat']['deter'][i][
                                                                     j - 1] -
                                                                 experiment['explanations']['obsfeat']['deter'][0][
                                                                     i + j]) ** 2)
                                expl_row["logit_mse"] = np.mean((experiment['explanations']['imgfeat']['logit'][i][
                                                                     j - 1] -
                                                                 experiment['explanations']['obsfeat']['logit'][0][
                                                                     i + j]) ** 2)
                                expl_row["stoch_distance"] = np.sum(
                                    experiment["explanations"]['imgfeat']['stoch'][i][j - 1].flatten() !=
                                    experiment['explanations']['obsfeat']['stoch'][0][i + j].flatten())
                                expl_row["image_lpips"] = lpips_distance(experiment['explanations']['image'][i][j],
                                                                         experiment['recorded_obs']["image"][0][i + j])
                                expl_row["rew_diff"] = abs(experiment["explanations"]["reward"][i][j - 1] -
                                                           experiment["recorded_obs"]["reward"][0][i + j])
                                expl_row["rew_sum"] = sum(experiment["explanations"]["reward"][i][:j - 1])
                                expl_row['rew_sum_diff'] = abs(
                                    sum(experiment["explanations"]["reward"][i][:j - 1]) - sum(
                                        experiment["recorded_obs"]["reward"][0][:i + j]))
                            except IndexError as e:
                                pass
                            expl_row["value_diff"] = abs(experiment["explanations"]["val"][i][j - 1] - sum(
                                experiment["recorded_obs"]["reward"][0][j - 1:]))

                        else:
                            expl_row["deter"] = experiment['explanations']['imgfeat']['deter'][i][j - 1]
                            expl_row["logit"] = experiment["explanations"]['imgfeat']['logit'][i][j - 1]
                            expl_row["stoch"] = experiment["explanations"]['imgfeat']['stoch'][i][j - 1]
                            expl_row["action"] = experiment["explanations"]["action"][i][j]
                            expl_row["action_logits"] = experiment['explanations']['logits'][i][j]
                            expl_row["action_probs"] = experiment['explanations']['probs'][i][j]
                            # shifted because we added the reconstruction to the beginning
                            expl_row["con"] = experiment["explanations"]["con"][i][j - 1]
                            expl_row["value"] = experiment["explanations"]["val"][i][j - 1]
                            expl_row["reward"] = experiment["explanations"]["reward"][i][j - 1]
                            try:
                                expl_row["image_mse"] = np.mean((experiment['explanations']['image'][i][j] -
                                                                 experiment['recorded_obs']["image"][0][i + j]) ** 2)
                                expl_row["deter_mse"] = np.mean((experiment['explanations']['imgfeat']['deter'][i][
                                                                     j - 1] -
                                                                 experiment['explanations']['obsfeat']['deter'][0][
                                                                     i + j]) ** 2)
                                expl_row["logit_mse"] = np.mean((experiment['explanations']['imgfeat']['logit'][i][
                                                                     j - 1] -
                                                                 experiment['explanations']['obsfeat']['logit'][0][
                                                                     i + j]) ** 2)
                                expl_row["stoch_distance"] = np.sum(
                                    experiment["explanations"]['obsfeat']['stoch'][0][i].flatten() !=
                                    experiment['explanations']['obsfeat']['stoch'][0][
                                        i + j].flatten())
                                expl_row["image_lpips"] = lpips_distance(experiment['explanations']['image'][i][j],
                                                                         experiment['recorded_obs']["image"][0][i + j])
                                expl_row["rew_diff"] = abs(experiment["explanations"]["val"][i][j - 1] -
                                                           experiment["recorded_obs"]["reward"][0][i + j])
                                expl_row["rew_sum"] = sum(experiment["explanations"]["reward"][i][:j - 1])
                                expl_row['rew_sum_diff'] = abs(
                                    sum(experiment["explanations"]["reward"][i][:j - 1]) - sum(
                                        experiment["recorded_obs"]["reward"][0][:i + j]))
                                expl_row["action_logits_mse"] = np.mean((experiment['explanations']['logits'][i][j] -
                                                                         experiment['recorded_outs']['logits'][0][
                                                                             i + j]) ** 2)
                                expl_row["action_probs_kl"] = entropy(
                                    experiment['recorded_outs']['probs'][0][i + j],
                                    experiment['explanations']['probs'][i][j])
                                expl_row["action_diff"] = not experiment["explanations"]["action"][i][j] == \
                                                              experiment["recorded_act"]["action"][0][i + j]
                            except IndexError as e:
                                pass
                            expl_row["value_diff"] = abs(experiment["explanations"]["val"][i][j - 1] - sum(
                                experiment["recorded_obs"]["reward"][0][j - 1:]))
                        expl_row.append()
            step_table.flush()
            explanation_table.flush()


def update_config(file_path, given_config, given_logdir, task, *overrides, size='size12m'):
    configs_str = elements.Path(file_path / 'dreamerv3/configs.yaml').read()
    configs = yaml.YAML(typ='safe').load(configs_str)

    config = elements.Config(configs['defaults'])
    config = config.update(configs[given_config])
    config = config.update({'replay_context': 0})
    logdir = elements.Path(given_logdir)
    config = config.update({'task': task})
    config = config.update(*overrides)

    if task.startswith('minigrid'):
        config = config.update({'task': task})
        config = config.update(configs[size])
    if task.startswith('highway'):
        config = config.update({'task': task})

    config = config.update({'run.from_checkpoint': logdir})

    return config


def make_agent(config):
    from dreamerv3.agent import Agent
    env = make_env(config, 0)
    notlog = lambda k: not k.startswith('log/')
    obs_space = {k: v for k, v in env.obs_space.items() if notlog(k)}
    act_space = {k: v for k, v in env.act_space.items() if k != 'reset'}
    env.close()
    return Agent(obs_space, act_space, elements.Config(
        **config.agent,
        logdir=config.logdir,
        seed=config.seed,
        jax=config.jax,
        batch_size=config.batch_size,
        batch_length=config.batch_length,
        replay_context=config.replay_context,
        report_length=config.report_length,
        replica=config.replica,
        replicas=config.replicas,
    ))


def make_env(config, index, **overrides):
    suite, task = config.task.split('_', 1)
    if suite == 'memmaze':
        from embodied.envs import from_gym
        import memory_maze  # noqa
    ctor = {
        'dummy': 'embodied.envs.dummy:Dummy',
        'gym': 'embodied.envs.from_gym:FromGym',
        'dm': 'embodied.envs.from_dmenv:FromDM',
        'crafter': 'embodied.envs.crafter:Crafter',
        'dmc': 'embodied.envs.dmc:DMC',
        'atari': 'embodied.envs.atari:Atari',
        'atari100k': 'embodied.envs.atari:Atari',
        'dmlab': 'embodied.envs.dmlab:DMLab',
        'minecraft': 'embodied.envs.minecraft:Minecraft',
        'loconav': 'embodied.envs.loconav:LocoNav',
        'pinpad': 'embodied.envs.pinpad:PinPad',
        'langroom': 'embodied.envs.langroom:LangRoom',
        'procgen': 'embodied.envs.procgen:ProcGen',
        'bsuite': 'embodied.envs.bsuite:BSuite',
        "minigrid": "embodied.envs.minigrid:Minigrid",
        "highway": "embodied.envs.highway:Highway",
        'memmaze': lambda task, **kw: from_gym.FromGym(
            f'MemoryMaze-{task}-v0', **kw),
    }[suite]
    if isinstance(ctor, str):
        module, cls = ctor.split(':')
        module = importlib.import_module(module)
        ctor = getattr(module, cls)
    kwargs = config.env.get(suite, {})
    kwargs.update(overrides)
    if kwargs.pop('use_seed', False):
        kwargs['seed'] = hash((config.seed, index)) % (2 ** 32 - 1)
    if kwargs.pop('use_logdir', False):
        kwargs['logdir'] = elements.Path(config.logdir) / f'env{index}'
    env = ctor(task, **kwargs)
    return wrap_env(env, config)


def load_checkpoint(agent, checkpoint_path):
    cp = elements.Checkpoint()
    cp.agent = agent
    cp.load(checkpoint_path, keys=['agent'])


def load_agent(file_path, given_logdir, given_config, task, checkpoint_path, size='size12m'):
    config = update_config(file_path, given_config, given_logdir, task, size=size)

    agent = make_agent(config)

    env = make_env(config, 0)
    act_space = env.act_space

    load_checkpoint(agent, checkpoint_path)

    return agent, env, act_space


def update_recorded(recorded, new_data):
    if recorded is None:
        return {k: v.copy() for k, v in new_data.items()}
    return {k: np.concatenate((recorded[k], new_data[k]), axis=0)
            for k, v in new_data.items()}


def add_gaussian_noise(image, sigma):
    noise = np.random.normal(0, sigma, image.shape)
    noisy_image = image + np.uint8(noise)
    return np.clip(noisy_image, 0, 255)


def collect_run_data(agent, env, act_space, obs_perturbation=None):
    acts = {k: [np.zeros(v.shape, v.dtype)] for k, v in act_space.items()}
    acts["reset"] = True
    carry = agent.init_policy(1) if agent.init_policy else None

    recorded_obs, recorded_act, recorded_outs = None, None, None
    steps = []
    consec = []
    step = 0

    while True:
        acts["action"] = acts["action"][0]
        obs_step = env.step(acts)
        if obs_perturbation is not None:
            obs_step['image'] = add_gaussian_noise(obs_step['image'], obs_perturbation)
        steps.append(step)
        consec.append(0 if step == 0 else 1)

        obs = {k: np.array([obs_step[k]]) for k in obs_step if not k.startswith("log/")}
        assert all(len(v) == 1 for v in obs.values()), obs

        carry, acts, outs, seed = agent.policy(carry, obs, mode='explain')

        if obs["is_last"][0]:
            acts["action"] = [-1]

        recorded_obs = update_recorded(recorded_obs, obs)
        recorded_act = update_recorded(recorded_act, acts)
        recorded_outs = update_recorded(recorded_outs, outs)

        acts["reset"] = obs["is_last"][0].copy()

        step += 1

        if obs["is_last"][0]:
            break

    for d in (recorded_obs, recorded_act, recorded_outs):
        for k in d:
            d[k] = np.expand_dims(d[k], axis=0)

    data = {
        **recorded_act,
        "consec": np.array([consec]),
        **recorded_obs,
        "stepid": np.array([steps]),
        "seed": seed,
    }
    return carry, data, recorded_obs, recorded_act, recorded_outs


def generate_explanation(agent, carry, data, horizon):
    carry, explanation, action = agent.explain(carry, data, horizon)
    # Combine explanation and action dictionaries and process them.
    explanation_combined = explanation | action
    explanation_combined = process_explanation(explanation_combined)
    return carry, explanation_combined, action


def start_run(agent, env, act_space, horizon: int = 15):
    carry, data, recorded_obs, recorded_act, recorded_outs = collect_run_data(agent, env, act_space)
    carry, explanation_combined, action = generate_explanation(agent, carry, data, horizon)
    return recorded_obs, recorded_act, recorded_outs, explanation_combined, action


def process_explanation(e, threshold):
    for key, values in e.items():
        if isinstance(e[key], np.ndarray):
            e[key] = list(e[key])
    for key, values in e['imgfeat'].items():
        if isinstance(e['imgfeat'][key], np.ndarray):
            e['imgfeat'][key] = list(e['imgfeat'][key])
    for i in range(len(e["con"])):
        for j in range(len(e["con"][i])):
            c = 1 - e["con"][i][j]
            if c > threshold:
                for key in e.keys():
                    if key == "image":
                        e[key][i] = e[key][i][: j + 2]
                    elif key in ['image-recon', 'obsfeat']:
                        pass
                    elif key in ['imgfeat']:
                        for featkey in e[key].keys():
                            e[key][featkey][i] = e[key][featkey][i][: j + 1]
                    else:
                        e[key][i] = e[key][i][: j + 1]
                break

    return e


def lpips_distance(image1, image2):
    img1 = Image.fromarray(image1, "RGB")
    img2 = Image.fromarray(image2, "RGB")
    img1_tensor = transform(img1).unsqueeze(0)
    img2_tensor = transform(img2).unsqueeze(0)
    with torch.no_grad():
        distance = loss_fn(img1_tensor, img2_tensor)

    return distance.item()


def get_subdirectories(path):
    return [path + f.name for f in pathlib.Path(path).iterdir() if f.is_dir()]

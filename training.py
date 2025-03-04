import pathlib
import warnings
from datetime import datetime
from functools import partial as bind

import elements
import ruamel.yaml as yaml

import embodied
from dreamerv3.explainer import experiments_to_hdf5, get_subdirectories, start_run
from dreamerv3.main import make_env, make_agent
from dreamerv3.main import make_replay, make_stream, make_logger


def main():
    warnings.simplefilter('ignore')
    folder = pathlib.Path.cwd()
    
    benchmarks = {
        'minigrid_SimpleCrossingS9N1': 'evaluation/checkpoints/test/minigrid_SimpleCrossingS9N1',
        'minigrid_SimpleCrossingS11N5': 'evaluation/checkpoints/test/minigrid_SimpleCrossingS11N5',
        'minigrid_LavaCrossingS9N1': 'evaluation/checkpoints/test/minigrid_LavaCrossingS9N1',
        'minigrid_DoorKey-5x5': 'evaluation/checkpoints/test/minigrid_DoorKey-5x5',
        'minigrid_DoorKey-16x16': 'evaluation/checkpoints/test/minigrid_DoorKey-16x16',
    }
    for k, v in benchmarks.items():
        size = 'size12m'
        configs = elements.Path(folder / 'dreamerv3/configs.yaml').read()
        configs = yaml.YAML(typ='safe').load(configs)
        config = elements.Config(configs['defaults'])
        config = config.update(configs[k.split('_')[0]])
        config = config.update({'seed': 530502})
        logdir = elements.Path(v)
        config = config.update({'logdir': logdir})
        config = config.update({'task': k})
        config = config.update(configs[size])
        logdir.mkdir()
        config.save(logdir / 'config.yaml')
        args = elements.Config(
            **config.run,
            replica=config.replica,
            replicas=config.replicas,
            logdir=config.logdir,
            batch_size=config.batch_size,
            batch_length=config.batch_length,
            report_length=config.report_length,
            consec_train=config.consec_train,
            consec_report=config.consec_report,
            replay_context=config.replay_context,
        )

        embodied.run.train_eval(
                bind(make_agent, config),
                bind(make_replay, config, 'replay'),
                bind(make_replay, config, 'eval_replay', 'eval'),
                bind(make_env, config),
                bind(make_env, config),
                bind(make_stream, config),
                bind(make_logger, config),
                args)

if __name__ == '__main__':
    main()
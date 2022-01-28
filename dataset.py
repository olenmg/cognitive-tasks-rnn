import gym
import neurogym as ngym
from neurogym.wrappers import ScheduleEnvs
from neurogym.utils.scheduler import RandomSchedule


def load_dataset(
    hp,
    task='random'
):
    """
    Args:
        hp: set of hyperparameters
        task: Simulation task
            'random', 'batchwise-random', task of yang19
    Return:
        env, dataset of given task
    """
    if task not in ['random', 'batchwise-random'] \
        and task not in ngym.get_collection('yang19'):
        raise ValueError(f"Cannot find the task: {task}")

    env = None
    if task == 'random':
        tasks = ngym.get_collection('yang19')
        env = ScheduleEnvs(
            envs=[gym.make(task, dt=hp['dt'], dim_ring=hp['dim_ring']) for task in tasks],
            schedule=RandomSchedule(len(tasks)),
            env_input=True
        )
    elif task == 'batchwise-random':
        # Experimental setting of Yang, et al.
        raise NotImplementedError
    else:
        env = gym.make(task, dt=hp['dt'], dim_ring=hp['dim_ring'])
    
    dataset = ngym.Dataset(
        env,
        batch_size=hp['batch_size'],
        seq_len=hp['seq_len']
    )
    
    return env, dataset

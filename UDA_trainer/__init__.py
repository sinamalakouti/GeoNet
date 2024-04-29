### __init__.py
# Function to get different types of adaptation trainings.
# Author: Tarun Kalluri @ 07/22
###
from . import mmd_global_trainer
from .cdan import train_cdan
from .mmd import train_mmd
from .plain import train_plain
from .dann import train_dann
from .val import val
from .eval import eval

def get_trainer(cfg):
    trainer = _get_trainer_instance(cfg['trainer'])
    return trainer


def _get_trainer_instance(name):
    try:
        return {
            'plain' : train_plain,
            'cdan' : train_cdan,
            'dann' : train_dann,
            'mmd' : train_mmd,
            'mmd_global_trainer': mmd_global_trainer
        }[name]
    except:
        raise BaseException('Trainer type {} not available'.format(name))



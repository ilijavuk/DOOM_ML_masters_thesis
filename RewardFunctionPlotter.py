from tensorboard.backend.event_processing import event_accumulator
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

event_file = './logs/logs_for_deathmatch\PPO_18'
event_acc = EventAccumulator(event_file)
event_acc.Reload()
scalar_keys = set()
scalar_keys.update(event_acc.Tags()['scalars'])
    
print(scalar_keys)

from collections import deque
from functools import partial
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from torch import nn
import typing as t

class Hook:
    """Wrapper for PyTorch forward hook mechanism."""
    def __init__(self, module: nn.Module, func: t.Callable):
        self.hook = None            # PyTorch's hook.
        self.module = module        # PyTorch layer to which the hook is attached to.
        self.func = func            # Function to call on each forward pass.
        self.register()

    def register(self):
        self.activation_data = deque(maxlen=1024)
        self.hook = self.module.register_forward_hook(partial(self.func, self))

    def remove(self):
        self.hook.remove()

def store_activation(hook, module, inp, outp):
    """Function intented to be called by a hook on a forward pass.
    
    Args:
        hook:    The hook object that generated the call.
        module:  The module on which the hook is registered.
        inp:     Input of the module.
        outp:    Output of the module.
    """
    hook.activation_data.append(outp.data.cpu().numpy())

from stable_baselines3.common.callbacks import BaseCallback


def get_low_act(data, threshold=0.2):
    """Computes the proportion of activations that have value close to zero."""
    low_activation = ((-threshold <= data) & (data <= threshold))
    return np.count_nonzero(low_activation) / np.size(low_activation)


# Callback for periodic logging to tensorboard.
class LayerActivationMonitoring(BaseCallback):
    
    def _on_rollout_start(self) -> None:
        """Called after the training phase."""
        
        hooks = self.model.policy.features_extractor.hooks
        
        # Remove the hooks so that they don't get called for rollout collection.
        for h in hooks: h.remove() 

        # Log last datapoint and statistics to tensorboard.
        for i, hook in enumerate(hooks):
            if len(hook.activation_data) > 0:
                data = hook.activation_data[-1]
                self.logger.record(f'diagnostics/activation_l{i}', data)
                self.logger.record(f'diagnostics/mean_l{i}', np.mean(data))
                self.logger.record(f'diagnostics/std_l{i}', np.std(data))
                self.logger.record(f'diagnostics/low_act_prop_l{i}', get_low_act(data))

    def _on_rollout_end(self) -> None:
        """Called before the training phase."""
        for h in self.model.policy.features_extractor.hooks: h.register()

    def _on_step(self):
        pass

def register_hooks(model):
    model.policy.features_extractor.hooks = [
        Hook(layer, store_activation)
        for layer in model.policy.features_extractor.cnn
        if isinstance(layer, nn.ReLU) or isinstance(layer, nn.LeakyReLU)]

def plot_activations(hooks):
    f = plt.figure(constrained_layout=False, figsize=(12, 8))
    gs = f.add_gridspec(3, 3)

    ax = [f.add_subplot(gs[0, :2]), f.add_subplot(gs[1, :2]), f.add_subplot(gs[2, :2])]
    ax_hists = [f.add_subplot(gs[0, 2]), f.add_subplot(gs[1, 2]), f.add_subplot(gs[2, 2])]

    ax[0].set_title('Layer activation mean')
    ax[1].set_title('Layer activation standard deviation')
    ax[2].set_title('Low activation proportion')

    for i, h in enumerate(hooks):
        activation_data = np.array(h.activation_data)
        stacked_data = np.stack(activation_data)
        
        # After stacking the data
        print(f"Stacked data shape: {stacked_data.shape}")

        # Before computing statistics
        print(f"Shape before statistics: {stacked_data.shape}")
        means = np.mean(stacked_data, axis=(1, 2, 3, 4))
        stds = np.std(stacked_data, axis=(1, 2, 3, 4))
        low_act = ((-0.2 <= stacked_data) & (stacked_data <= 0.2))
        low_act = np.count_nonzero(low_act, axis=(1, 2, 3, 4)) / np.prod(low_act.shape[1:])
        print(f"Means shape: {means.shape}")
        print(f"Stds shape: {stds.shape}")
        print(f"Low activation shape: {low_act.shape}")

        # Histograms
        bins = np.linspace(-7, 7, 40)
        melted_data = stacked_data.reshape(stacked_data.shape[0], -1)
        hist_img = np.apply_along_axis(
            lambda a: np.log1p(np.histogram(a, bins=bins)[0][::-1]), 1, melted_data)

        # Plot
        ax[0].plot(means, label=f'Mean layer {i}')
        ax[1].plot(stds, label=f'Std layer {i}')
        ax[2].plot(low_act, label=f'Low activation layer {i}')
        ax_hists[i].imshow(hist_img.T, aspect='auto')
        ax_hists[i].set_title(f'Activation histogram layer {i}')

    ax[0].set_ylim((-0.5, 0.5))
    ax[1].set_ylim((0, 1))
    ax[2].set_ylim((0, 1))

    for a in ax:
        a.grid(True)
        a.legend()

    plt.tight_layout()

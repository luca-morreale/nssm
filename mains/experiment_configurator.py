
import logging
import torch

import models
import datasets
import optimizers
import tasks
import loggers
import losses
import schedulers


class ExperimentConfigurator():

    def __init__(self):
        configurator = {}
        configurator['dataset']   = datasets.create
        configurator['model']     = models.create
        configurator['optimizers'] = optimizers.create
        configurator['schedulers'] = schedulers.create
        configurator['loss']       = losses.create
        configurator['tasks']      = tasks.create
        configurator['logging']    = loggers.create
        configurator['loop']       = lambda a,b : a
        self.configurator = configurator


    def create_experiment_modules(self, config):

        experiment = {}

        for key, value in sorted(config.items()):
            if key not in self.configurator:
                logging.info(f'Skipping key {key}')
                continue
            experiment[key] = self.configurator[key](value, experiment)

        if ('load_checkpoint', True) in config['model'].items():
            self.load_checkpoint(experiment, config['model']['checkpoint'])

        return experiment

    def load_checkpoint(self, experiment, ckpt_path):
        experiment['model'].load_state_dict(torch.load(ckpt_path))
        logging.info('Loaded checkpoint model')

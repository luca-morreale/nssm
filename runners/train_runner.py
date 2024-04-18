
from .generic_runner import GenericRunner

class TrainRunner(GenericRunner):

    ## base function for training
    def run(self, batch, model, experiment):

        ## call model and compute main losses
        model_out = self.forward_model(batch, model, experiment)
        ## compute regularization terms if any
        loss, logs = self.compute_losses(model, experiment, model_out, batch)

        return loss, logs

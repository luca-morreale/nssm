
import logging
from tqdm import trange
from collections import deque

from .training_loop import TrainingLoop


class GradientTrainingLoop(TrainingLoop):
    ### used in Neural Surface Maps
    losses = deque([])


    def should_stop(self, loss):
        ## compute the median gradient for all parameters (median of mean, easy aggregation and does not clutter logging)

        self.losses.append(loss)
        if len(self.losses) > 1000:
            self.losses.popleft()
        avg_loss = sum(self.losses) / len(self.losses)

        return avg_loss < self.grad_stop


    def loop(self):

        num_samples = len(self.train_loader)

        ## training loop
        for epoch in trange(self.num_epochs):

            grad = 0
            for batch in self.train_loader:

                converged = False

                self.zero_grads()

                batch = self.runner.move_to_device(batch)
                loss, logs = self.runner.train_step(batch)

                loss.backward()

                self.optimize()

                self.log_train(logs)

                ## check if the loss has vanished or below
                ## threshold then model has converged and can stop
                if self.should_stop(loss):
                    converged = True
                    break

            self.checkpointing(epoch)

            self.scheduling()

            if converged:
                logging.info('Stopping!! Model has low gradient')
                break

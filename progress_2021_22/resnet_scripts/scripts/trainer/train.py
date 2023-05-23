import torch

from os.path import abspath, dirname, join
from torch.utils.data.dataloader import DataLoader as DataLoader


class Trainer:
    def __init__(
        self,
        data_loader: DataLoader,
        model: torch.nn.Module,
        optimizer,
        loss_fn,
        device: torch.device = None,
        output_dir=None,
        log_interval: int = 100,
        lr_scheduler=None,
        callbacks=None,
    ):
        self._data_loader = data_loader
        self._model = model.to(device)
        self._device = device
        self._optimizer = optimizer
        self._loss_fn = loss_fn
        self._lr_scheduler = lr_scheduler
        self._output_dir = output_dir
        self._log_interval = log_interval

        self._global_step = 0

        if callbacks is not None:
            self._callbacks = callbacks.copy()

            # Set Callback properties
            for callback in self._callbacks:
                callback.model = self._model
                callback.optimizer = self._optimizer
                callback.lr_scheduler = self._lr_scheduler
                callback.batch_size = self._data_loader.batch_size
        else:
            self._callbacks = None

    def _check_property(self, prop_name: str):
        """Check that a property has been set.

        attr_name: Name of attribute to check.
        """
        if not hasattr(self, prop_name):
            raise RuntimeError(f"{prop_name} needs to be set in {type(self).__name__}")

    @property
    def output_dir(self):
        self._check_property("_output_dir")
        return self._output_dir

    @property
    def global_step(self):
        return self._global_step

    def fit_model(
        self,
        epochs: int,
    ):
        """Train the model.

        args:
            epochs: Number of epochs per batch
        """
        self._global_step = 0
        total_steps = epochs * len(self._data_loader)

        # start_before()
        for epoch in range(epochs):
            runnning_loss = 0.0

            for batch_step, batch in enumerate(self._data_loader):
                #for windowed_X, windowed_y in batch:
                #X, y = batch[0].to(self._device), batch[1].to(self._device)
                print(type(batch[0]), type(batch[1]))
                X = torch.tensor(batch[0]).to(self._device)
                y = torch.tensor(batch[1]).to(self._device)
                print(type(batch[0]), type(batch[1]))
                #exit(1)

                # Zero the parameter gradients
                self._optimizer.zero_grad()

                # Forward + Backward + Optimize
                outputs = self._model(X)
                loss = self._loss_fn(outputs, y)
                loss.backward()
                self._optimizer.step()

                curr_loss = loss.item()
                if self._callbacks is not None:
                    _, predicted = torch.max(outputs.data, 1)
                    for callback in self._callbacks:
                        callback(
                            global_step=self._global_step,
                            total_steps=total_steps,
                            X=X,
                            y=y,
                            loss=curr_loss,
                            avg_loss=runnning_loss / (batch_step + 1),
                            predicted=predicted,
                        )
                if self._lr_scheduler is not None:
                    self._lr_scheduler.step()

                runnning_loss += curr_loss
                self._global_step += 1

    def save_model(self):
        """Save the feature vector."""
        if self._output_dir is not None:
            torch.save(
                self._model(),
                join(dirname(dirname(dirname(abspath(__file__)))), self._output_dir),
            )

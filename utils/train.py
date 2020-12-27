import os
import torch


class TrainClass:

    def __init__(self, model, loss_fn, optimizer, metric, train_data,
                 val_data=None, epochs=10, print_every=500,
                 save_best=True, save_path="models/weights"):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.metric = metric
        self.train_data = train_data
        self.val_data = val_data
        self.epochs = epochs
        self.print_every = min(print_every, len(train_data))
        self.save_best = save_best
        self.save_path = save_path

    def train(self):
        # save losses for printing
        losses = []
        iteration = 0
        best_metric = 0
        for epoch in range(self.epochs):
            self.model.train()
            running_loss = 0.0
            for i in range(len(self.train_data)):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = self.train_data[i]

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, labels)
                loss.backward()
                self.optimizer.step()
                losses.append(loss.item())

                running_loss += loss.item()
                if i % self.print_every == self.print_every - 1:
                    iteration += self.print_every
                    print('Training loss iter %d loss: %.3f' %
                          (iteration, running_loss / self.print_every))
                    running_loss = 0.0
                    if self.val_data:
                        with torch.no_grad():
                            self.model.eval()
                            metric = self.metric(self.model, self.val_data)
                            if metric > best_metric:
                                best_metric = metric
                                print("Saving best model at %1.3f" % best_metric)
                                torch.save(self.model.state_dict(), os.path.join(self.save_path, "best_model_uncalibrated.pt"))
                            print("\t")
                            print("Validation metric: %1.3f" % metric)
        print('Finished Training')
        return losses

from module import AlphaZeroTrainer

from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from functools import partial
import numpy as np
import random

def train(config, checkpoint_dir=None):
    trainer = AlphaZeroTrainer(
        config['batch_size'],
        config['handouts'],
        config['rollouts'],
        config['alpha'],
        9,
        10,
        "module/training/tuner/data",
        "module/training/tuner",
        config['lr'],
        config['cput'],
        config['residual_layers'],
        config['num_filters'],
        config['tau_threshold'],
    )

    for e in range(15):
        data = trainer._get_data(True, 1, 1, True, trainer.batch_size)
        total_loss = 0
        batch_size = len(data)
        total = 0

        for _ in range(batch_size // config["sample"]):
            batch = random.sample(data, config["sample"])
            total += config["sample"]
            loss = trainer.net.train_batch(batch)
            total_loss += loss[0]
        
        loss = total_loss / total
        print(f'Loss at epoch {e + 1}: {loss}')
        tune.report(loss=loss)
    print("Finished Training epoch")


def main(num_samples, max_num_epochs, gpus_per_trial):
    config = {
        "batch_size": tune.sample_from(lambda _: 2 ** np.random.randint(6, 11)),
        "handouts": tune.choice([50, 100]),
        "rollouts": tune.choice([5, 10]),
        "alpha": tune.loguniform(0.8, 2),
        "tau_threshold": tune.sample_from(lambda _: np.random.randint(6, 12)),
        "lr": tune.loguniform(1e-6, 1e-4),
        "cput": tune.sample_from(lambda _: np.random.randint(1, 5)),
        "residual_layers": 8,
        "num_filters": 70,
        "sample": 35,
    }

    scheduler = ASHAScheduler(
        time_attr="training_iteration",
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=3,
        reduction_factor=2)

    reporter = CLIReporter(
        metric_columns=["loss", "training_iteration"])

    result = tune.run(
        train,
        resources_per_trial={"cpu": 2, "gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter)

    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))


if __name__ == "__main__":
    # You can change the number of GPUs per trial here:
    main(num_samples=20, max_num_epochs=15, gpus_per_trial=1)
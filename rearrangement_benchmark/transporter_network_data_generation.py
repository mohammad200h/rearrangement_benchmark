"""Generating data for the transporter network."""

from rearrangement_benchmark.rearrangement_task import RearrangementTask


from hydra import compose, initialize

TRANSPORTER_CONFIG = compose(config_name="transporter_data_collection")

if __name__=="__main__":
    task = RearrangementTask(cfg = TRANSPORTER_CONFIG)
    obs = task.reset()
    print(obs)


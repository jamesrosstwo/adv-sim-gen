import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from definitions import ROOT_PATH
from experiment import Experiment
from models.ppo import PPOPolicy


class TrainBaselineExperiment(Experiment):
    def __init__(
            self,
            policy: DictConfig,
            n_timesteps: int = 1_000_000,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self._ppo: PPOPolicy = PPOPolicy.from_conf(policy, env=self._env)
        self._n_timesteps = n_timesteps

    def run(self):
        self._ppo.learn(total_timesteps=self._n_timesteps)
        self._ppo.save(self._out_path / "ppo_car_racing")

@hydra.main(config_name="train_baseline", config_path=str(ROOT_PATH / "config"))
def main(cfg: DictConfig):
    exp = TrainBaselineExperiment(**cfg)
    exp.run()

if __name__ == "__main__":
    main()
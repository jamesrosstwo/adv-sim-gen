import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from definitions import ROOT_PATH
from experiment import Experiment
from policy.ppo import PPOPolicy


from imitation.algorithms.adversarial.gail import GAIL
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.policies.serialize import load_policy
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util.networks import RunningNorm
from imitation.util.util import make_vec_env


class TrainGAILExperiment(Experiment):
    def __init__(
            self,
            policy: DictConfig,
            n_timesteps: int = 1_000_000,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self._expert: PPOPolicy = instantiate(policy, env=self._env)
        self._n_timesteps = n_timesteps

    def run(self):
        self._ppo.learn(total_timesteps=self._n_timesteps)
        self._ppo.save(self._out_path / "ppo_car_racing")

@hydra.main(config_name="train_baseline", config_path=str(ROOT_PATH / "config"))
def main(cfg: DictConfig):
    exp = TrainGAILExperiment(**cfg)
    exp.run()

if __name__ == "__main__":
    main()
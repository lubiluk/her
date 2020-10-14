# Algorithms
from spinup.algos.pytorch.ddpg.ddpg import ddpg as ddpg_pytorch
from spinup.algos.pytorch.sac.sac import sac as sac_pytorch
from spinup.algos.pytorch.td3.td3 import td3 as td3_pytorch
from spinup.algos.pytorch.trpo.trpo import trpo as trpo_pytorch
from spinup.algos.pytorch.ddpg_her.ddpg_her import ddpg_her as ddpg_her_pytorch

# Loggers
from spinup.utils.logx import Logger, EpochLogger

# Version
from spinup.version import __version__
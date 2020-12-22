from model import BoidFlockers
from misc.runner import ModelRunner

param_path = "./parameter/nominal.toml"
runner = ModelRunner(BoidFlockers, param_path)
runner.run_silent_wo_plt()

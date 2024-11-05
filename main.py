import os
import hydra
from transformers import set_seed
from models.train_module import Trainer

import logging
log = logging.getLogger(__name__)

def get_config():
    try:
        path = os.environ["HYDRA_CONFIG_PATH"]
        name = os.environ["HYDRA_CONFIG_NAME"]
    except:
        path = os.path.dirname(os.environ["HYDRA_CONFIG_PATH"])
        name = os.path.basename(os.environ["HYDRA_CONFIG_PATH"])
    return path, name

@hydra.main(
    config_path=get_config()[0],
    config_name=get_config()[1],
)
def main(config):
    set_seed(config["seed"])
    os.environ["WANDB_WATCH"] = "False"
    auto_generated_dir = os.getcwd()
    log.info(f"Work dir: {auto_generated_dir}")
    os.chdir(hydra.utils.get_original_cwd())

    trainer = Trainer(config, auto_generated_dir)
    trainer.pipeline()

if __name__ == "__main__":
    main()
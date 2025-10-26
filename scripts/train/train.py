import os
import sys
sys.path.insert(0, './src')
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import hydra
import omegaconf
import pytorch_lightning as pl
import torch
import torch.multiprocessing
import wandb as wd
from omegaconf import OmegaConf, listconfig
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.utilities import rank_zero_only
from boltz.data.module.training import BoltzTrainingDataModule, DataConfig


@dataclass
class TrainConfig:
    """Train configuration.

    Attributes
    ----------
    data : DataConfig
        The data configuration.
    model : ModelConfig
        The model configuration.
    output : str
        The output directory.
    trainer : Optional[dict]
        The trainer configuration.
    resume : Optional[str]
        The resume checkpoint.
    pretrained : Optional[str]
        The pretrained model.
    wandb : Optional[dict]
        The wandb configuration.
    disable_checkpoint : bool
        Disable checkpoint.
    matmul_precision : Optional[str]
        The matmul precision.
    find_unused_parameters : Optional[bool]
        Find unused parameters.
    save_top_k : Optional[int]
        Save top k checkpoints.
    validation_only : bool
        Run validation only.
    debug : bool
        Debug mode.
    strict_loading : bool
        Fail on mismatched checkpoint weights.
    load_confidence_from_trunk: Optional[bool]
        Load pre-trained confidence weights from trunk.

    """

    data: DataConfig
    model: LightningModule
    output: str
    trainer: Optional[dict] = None
    resume: Optional[str] = None
    pretrained: Optional[str] = None
    wandb: Optional[dict] = None
    disable_checkpoint: bool = False
    matmul_precision: Optional[str] = None
    find_unused_parameters: Optional[bool] = False
    save_top_k: Optional[int] = 1
    every_n_train_steps: Optional[int] = 10
    validation_only: bool = False
    debug: bool = False
    strict_loading: bool = False
    load_confidence_from_trunk: Optional[bool] = False



# the first arg is the config file
def train(raw_config: str, args: list[str]) -> None:  # noqa: C901, PLR0912, PLR0915
    """Run training.

    Parameters
    ----------
    raw_config : str
        The input yaml configuration.
    args : list[str]
        Any command line overrides.

    """
    # Load the configuration
    raw_config = omegaconf.OmegaConf.load(raw_config) # load the config file

    # Apply input arguments
    args = omegaconf.OmegaConf.from_dotlist(args) # convert the args to a config
    raw_config = omegaconf.OmegaConf.merge(raw_config, args) # merge the config file and the args

    # Instantiate the task
    cfg = hydra.utils.instantiate(raw_config) # instantiate the config
    cfg = TrainConfig(**cfg) # convert the config to a TrainConfig object

    # Set matmul precision
    if cfg.matmul_precision is not None: # default is None
        torch.set_float32_matmul_precision(cfg.matmul_precision)

    # Create trainer dict
    trainer = cfg.trainer # load the trainer config
    if trainer is None:
        trainer = {}

    # Flip some arguments in debug mode
    devices = trainer.get("devices", 1)

    # wandb settings
    wandb = cfg.wandb # load the wandb config
    if cfg.debug: # debug mode
        """
        only use one gpu for debug mode
        """
        if isinstance(devices, int):
            devices = 1
        elif isinstance(devices, (list, listconfig.ListConfig)):
            devices = [devices[0]]  
        trainer["devices"] = devices # set the devices to 1
        cfg.data.num_workers = 0
        if wandb: # disable wandb in debug mode
            wandb = None

    print("Devices: ", devices)

    # Create objects
    data_config = DataConfig(**cfg.data) # convert the data config to a DataConfig object
    data_module = BoltzTrainingDataModule(data_config)
    model_module = cfg.model # load the model config

    if cfg.pretrained and not cfg.resume:
        # Load the pretrained weights into the confidence module
        if cfg.load_confidence_from_trunk:
            checkpoint = torch.load(cfg.pretrained, map_location="cpu")

            # Modify parameter names in the state_dict
            new_state_dict = {}
            for key, value in checkpoint["state_dict"].items():
                if not key.startswith("structure_module") and not key.startswith(
                    "distogram_module"
                ):
                    new_key = "confidence_module." + key
                    new_state_dict[new_key] = value
            new_state_dict.update(checkpoint["state_dict"])

            # Update the checkpoint with the new state_dict
            checkpoint["state_dict"] = new_state_dict
        else:
            file_path = cfg.pretrained

        print(f"Loading model from {file_path}")
        model_module = type(model_module).load_from_checkpoint(
            file_path, strict=False, 
            map_location=f"cpu",
            **(model_module.hparams)
        )

        if cfg.load_confidence_from_trunk:
            os.remove(file_path)

    # Create checkpoint callback
    callbacks = []
    dirpath = cfg.output
    if not cfg.disable_checkpoint:
        mc = ModelCheckpoint(
            monitor="train/loss",
            save_top_k=cfg.save_top_k,
            save_last=True,
            mode="min",
            every_n_train_steps=cfg.every_n_train_steps,
        )
        callbacks = [mc]

    # Create wandb logger
    loggers = []
    if wandb:

        wd.init(
            project=wandb["project"],
            entity=wandb["entity"],
            name=wandb["name"],
        )

        wdb_logger = WandbLogger(
            group=wandb["name"],
            save_dir=cfg.output,
            project=wandb["project"],
            entity=wandb["entity"],
            log_model=False,
        )
        loggers.append(wdb_logger)
        # Save the config to wandb

        @rank_zero_only
        def save_config_to_wandb() -> None:
            config_out = Path(wdb_logger.experiment.dir) / "run.yaml"
            with Path.open(config_out, "w") as f:
                OmegaConf.save(raw_config, f)
            wdb_logger.experiment.save(str(config_out))

        save_config_to_wandb()

    # Set up trainer
    strategy = "auto"
    if (isinstance(devices, int) and devices > 1) or (
        isinstance(devices, (list, listconfig.ListConfig)) and len(devices) > 1
    ):
        strategy = DDPStrategy(find_unused_parameters=cfg.find_unused_parameters)

    trainer = pl.Trainer(
        default_root_dir=str(dirpath),
        strategy=strategy,
        callbacks=callbacks,
        logger=loggers,
        enable_checkpointing=not cfg.disable_checkpoint,
        reload_dataloaders_every_n_epochs=1,
        **trainer,
    )

    if not cfg.strict_loading:
        model_module.strict_loading = False

    if cfg.validation_only:
        trainer.validate(
            model_module,
            datamodule=data_module,
            ckpt_path=cfg.resume,
        )
    else:
        trainer.fit(
            model_module,
            datamodule=data_module,
            ckpt_path=cfg.resume,
        )


if __name__ == "__main__":
    arg1 = sys.argv[1] # config file
    arg2 = sys.argv[2:] # other args
    train(arg1, arg2)

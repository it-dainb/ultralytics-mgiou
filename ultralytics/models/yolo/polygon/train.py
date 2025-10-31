# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from copy import copy
from pathlib import Path
from typing import Any

from ultralytics.models import yolo
from ultralytics.nn.tasks import PolygonModel
from ultralytics.utils import DEFAULT_CFG, LOGGER


class PolygonTrainer(yolo.detect.DetectionTrainer):
    """
    A class extending the DetectionTrainer class for training YOLO pose estimation models.

    This trainer specializes in handling pose estimation tasks, managing model training, validation, and visualization
    of pose keypoints alongside bounding boxes.

    Attributes:
        args (dict): Configuration arguments for training.
        model (PolygonModel): The polygon model being trained.
        data (dict): Dataset configuration including polygon shape information.
        loss_names (tuple): Names of the loss components used in training.

    Methods:
        get_model: Retrieve a polygon model with specified configuration.
        set_model_attributes: Set polygon shape attribute on the model.
        get_validator: Create a validator instance for model evaluation.
        plot_training_samples: Visualize training samples with polygons.
        get_dataset: Retrieve the dataset and ensure it contains required np key.

    Examples:
        >>> from ultralytics.models.yolo.pose import PoseTrainer
        >>> args = dict(model="yolo11n-pose.pt", data="coco8-pose.yaml", epochs=3)
        >>> trainer = PoseTrainer(overrides=args)
        >>> trainer.train()
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict[str, Any] | None = None, _callbacks=None):
        """
        Initialize a PoseTrainer object for training YOLO pose estimation models.

        Args:
            cfg (dict, optional): Default configuration dictionary containing training parameters.
            overrides (dict, optional): Dictionary of parameter overrides for the default configuration.
            _callbacks (list, optional): List of callback functions to be executed during training.

        Notes:
            This trainer will automatically set the task to 'pose' regardless of what is provided in overrides.
            A warning is issued when using Apple MPS device due to known bugs with pose models.
        """
        if overrides is None:
            overrides = {}
        overrides["task"] = "polygon"
        super().__init__(cfg, overrides, _callbacks)
        
        # Extract hybrid loss parameters from overrides
        self.use_mgiou = overrides.get("use_mgiou", False)
        self.use_hybrid = overrides.get("use_hybrid", False)
        self.alpha_schedule = overrides.get("alpha_schedule", "cosine")
        self.alpha_start = overrides.get("alpha_start", 0.9)
        self.alpha_end = overrides.get("alpha_end", 0.2)

        if isinstance(self.args.device, str) and self.args.device.lower() == "mps":
            LOGGER.warning(
                "Apple MPS known Pose bug. Recommend 'device=cpu' for Pose models. "
                "See https://github.com/ultralytics/ultralytics/issues/4031."
            )

    def get_model(
        self,
        cfg: str | Path | dict[str, Any] | None = None,
        weights: str | Path | None = None,
        verbose: bool = True,
    ) -> PolygonModel:
        """
        Get pose estimation model with specified configuration and weights.

        Args:
            cfg (str | Path | dict, optional): Model configuration file path or dictionary.
            weights (str | Path, optional): Path to the model weights file.
            verbose (bool): Whether to display model information.

        Returns:
            (PolygonModel): Initialized pose estimation model.
        """
        model = PolygonModel(
            cfg, 
            nc=self.data["nc"], 
            ch=self.data["channels"], 
            data_np=self.data["np"], 
            verbose=verbose, 
            use_mgiou=self.use_mgiou,
            use_hybrid=self.use_hybrid,
            alpha_schedule=self.alpha_schedule,
            alpha_start=self.alpha_start,
            alpha_end=self.alpha_end,
            total_epochs=self.epochs
        )
        if weights:
            model.load(weights)

        return model

    def set_model_attributes(self):
        """Set keypoints shape attribute of PolygonModel."""
        super().set_model_attributes()
        self.model.np = self.data["np"]

    def get_validator(self):
        """Return an instance of the PolygonValidator class for validation."""
        self.loss_names = "box_loss", "polygon_loss", "cls_loss", "dfl_loss", "mgiou_loss"
        return yolo.polygon.PolygonValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def get_dataset(self) -> dict[str, Any]:
        """
        Retrieve the dataset and add `np` from model config if not in dataset.

        Returns:
            (dict): A dictionary containing the training/validation/test dataset and category names.

        Notes:
            If `np` is not in the dataset YAML, it will be read from the model configuration.
            This allows the model YAML to define the number of polygon points.
        """
        from ultralytics.nn.tasks import yaml_model_load
        
        data = super().get_dataset()
        if "np" not in data:
            if not isinstance(self.args.model, dict):
                model_cfg = yaml_model_load(self.args.model)
            else:
                model_cfg = self.args.model
            
            if "np" in model_cfg:
                data["np"] = model_cfg["np"]
                LOGGER.info(f"Using np={model_cfg['np']} from model configuration")
            else:
                raise KeyError(
                    f"No `np` found in either {self.args.data} or {self.args.model}. "
                    f"Please specify the number of polygon points in your model or dataset YAML."
                )
        return data
    
    def on_train_epoch_start(self):
        """Called at the start of each training epoch to update loss function's epoch counter."""
        super().on_train_epoch_start() if hasattr(super(), 'on_train_epoch_start') else None
        
        # Update epoch in loss function for hybrid scheduling
        if hasattr(self.model, 'criterion') and hasattr(self.model.criterion, 'set_epoch'):
            self.model.criterion.set_epoch(self.epoch)



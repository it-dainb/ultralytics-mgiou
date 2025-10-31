# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import LOGGER, ops
from ultralytics.utils.metrics import OKS_SIGMA, PoseMetrics, kpt_iou, poly_iou


class PolygonValidator(DetectionValidator):
    """
    A class extending the DetectionValidator class for validation based on a polygon model.

    This validator is specifically designed for polygon detection tasks, handling polygon vertices and implementing
    specialized metrics for polygon evaluation.

    Attributes:
        sigma (np.ndarray): Sigma values for polygon IoU calculation (one value per vertex).
        np (int): Number of polygon vertices.
        args (dict): Arguments for the validator including task set to "polygon".
        metrics (PoseMetrics): Metrics object for polygon evaluation (reuses pose metrics infrastructure).

    Methods:
        preprocess: Preprocess batch by converting polygon data to float and moving it to the device.
        get_desc: Return description of evaluation metrics in string format.
        init_metrics: Initialize polygon detection metrics for YOLO model.
        _prepare_batch: Prepare a batch for processing by converting polygons to float and scaling to original
            dimensions.
        _prepare_pred: Prepare and scale polygon vertices in predictions for processing.
        _process_batch: Return correct prediction matrix by computing Intersection over Union (IoU) between
            polygon detections and ground truth.
        plot_val_samples: Plot and save validation set samples with ground truth bounding boxes and polygons.
        plot_predictions: Plot and save model predictions with bounding boxes and polygons.
        save_one_txt: Save YOLO polygon detections to a text file in normalized coordinates.
        pred_to_json: Convert YOLO predictions to COCO JSON format.
        eval_json: Evaluate polygon detection model using COCO JSON format.

    Examples:
        >>> from ultralytics.models.yolo.polygon import PolygonValidator
        >>> args = dict(model="yolo11n-polygon.pt", data="coco8-polygon.yaml")
        >>> validator = PolygonValidator(args=args)
        >>> validator()
    
    Notes:
        Internally uses "keypoints" field name for compatibility with inherited pose validation infrastructure,
        but the data represents polygon vertices (N, num_vertices, 2).
    """

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None) -> None:
        """
        Initialize a PolygonValidator object for polygon detection validation.

        This validator is specifically designed for polygon detection tasks, handling polygon vertices and implementing
        specialized metrics for polygon evaluation using MGIoU.

        Args:
            dataloader (torch.utils.data.DataLoader, optional): Dataloader to be used for validation.
            save_dir (Path | str, optional): Directory to save results.
            args (dict, optional): Arguments for the validator including task set to "polygon".
            _callbacks (list, optional): List of callback functions to be executed during validation.

        Examples:
            >>> from ultralytics.models.yolo.polygon import PolygonValidator
            >>> args = dict(model="yolo11n-polygon.pt", data="coco8-polygon.yaml")
            >>> validator = PolygonValidator(args=args)
            >>> validator()

        Notes:
            This class extends DetectionValidator with polygon-specific functionality. It reuses PoseMetrics
            infrastructure for evaluation. A warning is displayed when using Apple MPS due to a known bug.
        """
        super().__init__(dataloader, save_dir, args, _callbacks)
        self.sigma = None
        self.np = None
        self.args.task = "polygon"
        self.metrics = PoseMetrics()
        if isinstance(self.args.device, str) and self.args.device.lower() == "mps":
            LOGGER.warning(
                "Apple MPS known bug. Recommend 'device=cpu' for Polygon models. "
                "See https://github.com/ultralytics/ultralytics/issues/4031."
            )

    def get_dataloader(self, dataset_path: str, batch_size: int):
        """Construct and return dataloader, ensuring np field is populated in data dict."""
        dataloader = super().get_dataloader(dataset_path, batch_size)
        # Ensure np is in self.data for polygon validation
        if self.data and "np" not in self.data:
            self._populate_np_from_model()
        return dataloader

    def _populate_np_from_model(self):
        """Populate the np field in self.data from model configuration if missing."""
        from ultralytics.nn.tasks import yaml_model_load
        
        # Try to get np from model if we have it loaded
        if hasattr(self, 'model') and hasattr(self.model, 'np'):
            self.data["np"] = self.model.np
            LOGGER.info(f"Using np={self.model.np} from loaded model")
        # Try to get from model file
        elif self.args.model:
            try:
                if not isinstance(self.args.model, dict):
                    model_cfg = yaml_model_load(self.args.model)
                else:
                    model_cfg = self.args.model
                
                if "np" in model_cfg:
                    self.data["np"] = model_cfg["np"]
                    LOGGER.info(f"Using np={model_cfg['np']} from model configuration")
            except Exception as e:
                LOGGER.debug(f"Could not load np from model config: {e}")
        
        # If still not found, we'll handle it in init_metrics with a default

    def preprocess(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Preprocess batch by converting polygon data to float and moving it to the device."""
        batch = super().preprocess(batch)
        if "polygons" in batch:
            batch["polygons"] = batch["polygons"].float()
        return batch

    def get_desc(self) -> str:
        """Return description of evaluation metrics in string format."""
        return ("%22s" + "%11s" * 10) % (
            "Class",
            "Images",
            "Instances",
            "Box(P",
            "R",
            "mAP50",
            "mAP50-95)",
            "Polygon(P",
            "R",
            "mAP50",
            "mAP50-95)",
        )

    def init_metrics(self, model: torch.nn.Module) -> None:
        """
        Initialize evaluation metrics for YOLO polygon validation.

        Args:
            model (torch.nn.Module): Model to validate.
        """
        super().init_metrics(model)
        
        # Try to get np from data first, then from model, with a sensible default
        if "np" in self.data:
            self.np = self.data["np"]
        elif hasattr(model, "np"):
            self.np = model.np
        else:
            # Try to infer from model config
            if hasattr(model, "yaml") and "np" in model.yaml:
                self.np = model.yaml["np"]
            else:
                LOGGER.warning("Could not determine polygon vertex count (np). Using default value of 4.")
                self.np = 4
        
        # For polygon task, np is just an integer (number of points)
        # For pose task, it would be a list like [17, 3]
        is_pose = isinstance(self.np, list) and self.np == [17, 3]
        nkpt = self.np[0] if isinstance(self.np, list) else self.np
        self.sigma = OKS_SIGMA if is_pose else np.ones(nkpt) / nkpt

    def postprocess(self, preds: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Postprocess YOLO predictions to extract and reshape polygon vertices.

        This method extends the parent class postprocessing by extracting polygon vertices from the 'extra'
        field of predictions and reshaping them according to the polygon vertex count configuration.
        The vertices are reshaped from a flattened format to the proper dimensional structure
        (typically [N, num_vertices, 2] for polygon format).

        Args:
            preds (torch.Tensor): Raw prediction tensor from the YOLO polygon model containing
                bounding boxes, confidence scores, class predictions, and polygon vertex data.

        Returns:
            (dict[torch.Tensor]): Dict of processed prediction dictionaries, each containing:
                - 'bboxes': Bounding box coordinates
                - 'conf': Confidence scores
                - 'cls': Class predictions
                - 'keypoints': Reshaped polygon vertices with shape (-1, self.np, 2)

        Note:
            Internally stores vertices in 'keypoints' field for compatibility with PoseMetrics infrastructure.
            The 'extra' field contains polygon vertex data beyond basic detection outputs.
        """
        preds = super().postprocess(preds)
        for pred in preds:
            # For polygon: self.np is int (e.g., 4), reshape to (N, np, 2)
            # For pose: self.np is list [17, 3], reshape to (N, 17, 3)
            if isinstance(self.np, list):
                pred["keypoints"] = pred.pop("extra").view(-1, *self.np)
            else:
                pred["keypoints"] = pred.pop("extra").view(-1, self.np, 2)
        return preds

    def _prepare_batch(self, si: int, batch: dict[str, Any]) -> dict[str, Any]:
        """
        Prepare a batch for processing by converting polygon vertices to float and scaling to original dimensions.

        Args:
            si (int): Batch index.
            batch (dict[str, Any]): Dictionary containing batch data with keys like 'polygons', 'batch_idx', etc.

        Returns:
            (dict[str, Any]): Prepared batch with polygon vertices scaled to original image dimensions.

        Notes:
            This method extends the parent class's _prepare_batch method by adding polygon processing.
            Polygon vertices are scaled from normalized coordinates to original image dimensions.
            Internally stored as 'keypoints' for compatibility with PoseMetrics infrastructure.
        """
        pbatch = super()._prepare_batch(si, batch)
        polygons = batch["polygons"][batch["batch_idx"] == si]
        h, w = pbatch["imgsz"]
        polygons = polygons.clone()
        polygons[..., 0] *= w
        polygons[..., 1] *= h
        pbatch["keypoints"] = polygons  # Store as keypoints for compatibility with pose metrics
        return pbatch

    def _process_batch(self, preds: dict[str, torch.Tensor], batch: dict[str, Any]) -> dict[str, np.ndarray]:
        """
        Return correct prediction matrix by computing Intersection over Union (IoU) between detections and ground truth.

        Args:
            preds (dict[str, torch.Tensor]): Dictionary containing prediction data with keys 'cls' for class predictions
                and 'keypoints' for keypoint predictions.
            batch (dict[str, Any]): Dictionary containing ground truth data with keys 'cls' for class labels,
                'bboxes' for bounding boxes, and 'keypoints' for keypoint annotations.

        Returns:
            (dict[str, np.ndarray]): Dictionary containing the correct prediction matrix including 'tp_p' for polygon
                true positives across 10 IoU levels.

        Notes:
            Uses MGIoU-based poly_iou metric for polygon matching, which returns GIoU scores in range [-1, 1].
        """
        tp = super()._process_batch(preds, batch)
        gt_cls = batch["cls"]
        if gt_cls.shape[0] == 0 or preds["cls"].shape[0] == 0:
            tp_p = np.zeros((preds["cls"].shape[0], self.niou), dtype=bool)
        else:
            # Sanitize predictions to avoid Inf/NaN during validation
            # This can happen in early training when model is unstable
            pred_kpts = preds["keypoints"]
            if torch.isnan(pred_kpts).any() or torch.isinf(pred_kpts).any():
                LOGGER.warning(
                    f"Detected NaN/Inf in polygon predictions during validation. "
                    f"Replacing with safe values. This may indicate unstable training."
                )
                pred_kpts = torch.nan_to_num(pred_kpts, nan=0.0, posinf=1e6, neginf=-1e6)
            
            # Use polygon IoU instead of keypoint-based OKS
            # poly_iou returns GIoU-based scores in range [-1, 1]
            iou = poly_iou(batch["keypoints"], pred_kpts)
            tp_p = self.match_predictions(preds["cls"], gt_cls, iou).cpu().numpy()
        tp.update({"tp_p": tp_p})  # update tp with polygon IoU
        return tp

    def save_one_txt(self, predn: dict[str, torch.Tensor], save_conf: bool, shape: tuple[int, int], file: Path) -> None:
        """
        Save YOLO polygon detections to a text file in normalized coordinates.

        Args:
            predn (dict[str, torch.Tensor]): Dictionary containing predictions with keys 'bboxes', 'conf', 'cls' and 'keypoints'.
            save_conf (bool): Whether to save confidence scores.
            shape (tuple[int, int]): Shape of the original image (height, width).
            file (Path): Output file path to save detections.

        Notes:
            The output format is: class_id x_center y_center width height confidence vertices where vertices are
            normalized (x, y) coordinate pairs for each polygon vertex.
        """
        from ultralytics.engine.results import Results

        Results(
            np.zeros((shape[0], shape[1]), dtype=np.uint8),
            path=None,
            names=self.names,
            boxes=torch.cat([predn["bboxes"], predn["conf"].unsqueeze(-1), predn["cls"].unsqueeze(-1)], dim=1),
            keypoints=predn["keypoints"],
        ).save_txt(file, save_conf=save_conf)

    def pred_to_json(self, predn: dict[str, torch.Tensor], pbatch: dict[str, Any]) -> None:
        """
        Convert YOLO predictions to COCO JSON format.

        This method takes prediction tensors and a filename, converts the bounding boxes from YOLO format
        to COCO format, and appends the results to the internal JSON dictionary (self.jdict).

        Args:
            predn (dict[str, torch.Tensor]): Prediction dictionary containing 'bboxes', 'conf', 'cls',
                and 'keypoints' tensors.
            pbatch (dict[str, Any]): Batch dictionary containing 'imgsz', 'ori_shape', 'ratio_pad', and 'im_file'.

        Notes:
            The method extracts the image ID from the filename stem (either as an integer if numeric, or as a string),
            converts bounding boxes from xyxy to xywh format, and adjusts coordinates from center to top-left corner
            before saving to the JSON dictionary.
        """
        super().pred_to_json(predn, pbatch)
        kpts = predn["kpts"]
        for i, k in enumerate(kpts.flatten(1, 2).tolist()):
            self.jdict[-len(kpts) + i]["keypoints"] = k  # keypoints

    def scale_preds(self, predn: dict[str, torch.Tensor], pbatch: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Scales predictions to the original image size."""
        return {
            **super().scale_preds(predn, pbatch),
            "kpts": ops.scale_coords(
                pbatch["imgsz"],
                predn["keypoints"].clone(),
                pbatch["ori_shape"],
                ratio_pad=pbatch["ratio_pad"],
            ),
        }

    def eval_json(self, stats: dict[str, Any]) -> dict[str, Any]:
        """Evaluate object detection model using COCO JSON format."""
        anno_json = self.data["path"] / "annotations/person_keypoints_val2017.json"  # annotations
        pred_json = self.save_dir / "predictions.json"  # predictions
        return super().coco_evaluate(stats, pred_json, anno_json, ["bbox", "keypoints"], suffix=["Box", "Polygon"])

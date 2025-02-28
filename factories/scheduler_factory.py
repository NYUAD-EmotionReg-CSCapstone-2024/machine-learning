from torch.optim import lr_scheduler

from .base_factory import BaseFactory

class SchedulerFactory(BaseFactory):
    REGISTRY = {
        "cosine_warmup": {
            "scheduler": lr_scheduler.CosineAnnealingWarmRestarts,
            "mandatory_params": ["T_0", "eta_min"],
            "optional_params": ["T_mult"]
        }
    }
    ITEM_KEY = "scheduler"
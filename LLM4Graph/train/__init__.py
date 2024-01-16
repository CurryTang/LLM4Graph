from .single_gt import train as single_gpu_train
from .eval import single_gpu_test
from .utils import get_metric_from_cfg, get_loss_fn_from_cfg, get_optim_from_cfg, get_scheduler_from_cfg
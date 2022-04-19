import os

from pytorch_lightning.utilities import _module_available

EXAMPLES_ROOT = os.path.dirname(__file__)
PACKAGE_ROOT = os.path.dirname(EXAMPLES_ROOT)
DATASETS_PATH = os.path.join(PACKAGE_ROOT, 'Datasets')

TORCHVISION_AVAILABLE = _module_available("torchvision")
DALI_AVAILABLE = _module_available("nvidia.dali")


from yacs.config import CfgNode as CN

###########################
# Config definition
###########################

_C = CN()

_C.VERSION = 1

# Directory to save the output files (like log.txt and model weights)
_C.OUTPUT_DIR = "./output"
# Path to a directory where the files were saved previously
_C.RESUME = ""
# Set seed to negative value to randomize everything
# Set seed to positive value to use a fixed seed
_C.SEED = -1
_C.USE_CUDA = True
# Print detailed information
# E.g. trainer, dataset, and backbone
_C.VERBOSE = True

###########################
# Input
###########################
_C.INPUT = CN()
# _C.INPUT.SIZE = (224, 224)
_C.INPUT.SIZE = (32, 32)
# Mode of interpolation in resize functions
_C.INPUT.INTERPOLATION = "bilinear"
# For available choices please refer to transforms.py
_C.INPUT.TRANSFORMS = ()
# If True, tfm_train and tfm_test will be None
_C.INPUT.NO_TRANSFORM = False
# Mean and std (default: ImageNet)
# _C.INPUT.PIXEL_MEAN = [0.485, 0.456, 0.406]
# _C.INPUT.PIXEL_STD = [0.229, 0.224, 0.225]
_C.INPUT.PIXEL_MEAN = [0.5071, 0.4865, 0.4409]
_C.INPUT.PIXEL_STD = [0.2673, 0.2564, 0.2762]
# Random crop
_C.INPUT.CROP_PADDING = 4
# Random resized crop
_C.INPUT.RRCROP_SCALE = (0.08, 1.0)
# Cutout
_C.INPUT.CUTOUT_N = 1
_C.INPUT.CUTOUT_LEN = 16
# Gaussian noise
_C.INPUT.GN_MEAN = 0.0
_C.INPUT.GN_STD = 0.15
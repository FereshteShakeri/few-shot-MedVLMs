from .sicapv2 import SicapV2
from .nct import NCT
from .camelyonpatch import PCAM
from .skincancer import SKIN
from .lc_lung import LCLUNG
from .lc_colon import LCCOLON


dataset_list = {
                "sicapv2": SicapV2,
                "nct": NCT,
                "pcam": PCAM,
                "skincancer": SKIN,
                "lclung": LCLUNG,
                "lccolon": LCCOLON,
                }


def build_dataset(dataset, root_path, shots):
    return dataset_list[dataset](root_path, shots)

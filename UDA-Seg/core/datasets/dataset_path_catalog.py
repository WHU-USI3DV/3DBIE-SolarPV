import os
from .sh_wv2 import sh_wv2_Dataset
from .wh_wv2 import wh_wv2_Dataset
from .vegas_wv2 import vegas_wv2_Dataset
from .potsdamRGB import potsdamRGBDataSet
from .potsdamIRGB import potsdamIRGBDataSet
from .vaihingen import vaihingenDataSet
from .vaihingen_self_distill import vaihingenSelfDistillDataSet
from .JA_GES import JA_GES_Dataset
from .HS_GES import HS_GES_Dataset
from .HY_GES import HY_GES_Dataset
from .QK_TDT import QK_TDT_Dataset
from .WC_GES import WC_GES_Dataset
from .QS_GES import QS_GES_Dataset
from .QK_GES import QK_GES_Dataset
from .HS_Right_GES import HS_Right_GES_Dataset
from .HS_Middle_GES import HS_Middle_GES_Dataset
from .HS_Left_GES import HS_Left_GES_Dataset
from .DXH_AS_GES import DXH_AS_GES_Dataset
from .HN_AS_GES import HN_AS_GES_Dataset
from .JH_GES import JH_GES_Dataset
from .XZ_AS_GES import XZ_AS_GES_Dataset
from .CD_AS_GES import CD_AS_GES_Dataset
from .HP_AS_GES import HP_AS_GES_Dataset
from .JX_AS_GES import JX_AS_GES_Dataset
from .JA_GES_self_distill import JAGESSelfDistillDataSet


class DatasetCatalog(object):
    DATASET_DIR = "datasets"
    RS_DATASET_DIR = "/home/cz/data/wv2_datasets" # dir path for dataset
    ISPRS_DATASET_DIR = "/home/cz/data/ISPRS_datasets/data" # dir path for dataset
    WH_GES_TEST_DIR = "/home/cz/data/GES_IMG_TEST" # dir path for dataset
    DATASETS = {
        "gta5_train": {
            "data_dir": "gta5",
            "data_list": "gta5_train_list.txt"
        },
        "cityscapes_train": {
            "data_dir": "cityscapes",
            "data_list": "cityscapes_train_list.txt"
        },
        "cityscapes_self_distill_train": {
            "data_dir": "cityscapes",
            "data_list": "cityscapes_train_list.txt",
            "label_dir": "cityscapes/soft_labels/inference/cityscapes_train"
        },
        "cityscapes_val": {
            "data_dir": "cityscapes",
            "data_list": "cityscapes_val_list.txt"
        },
        "sh_wv2_train": {
            "data_dir": "sh_wv2",
            "data_list": "sh_wv2_train_list.txt"
        },
        "sh_wv2_test": {
            "data_dir": "sh_wv2",
            "data_list": "sh_wv2_test_list.txt"
        },
        "JA_GES_train": {
            "data_dir": "JA_GES",
            "data_list": "JA_GES_train_list.txt"
        },
        "JA_GES_test": {
            "data_dir": "JA_GES",
            "data_list": "JA_GES_test_list.txt"
        },
        "JA_GES_self_distill_train": {
            "data_dir": "JA_GES",
            "data_list": "JA_GES_train_list.txt",
            "label_dir": "soft_labels/JA_GES_train"
        },
        "HS_GES_train": {
            "data_dir": "HS_GES",
            "data_list": "HS_GES_train_list.txt"
        },
        "HS_GES_test": {
            "data_dir": "HS_GES",
            "data_list": "HS_GES_test_list.txt"
        },
        "HS_Right_GES_train": {
            "data_dir": "HS_Right_GES",
            "data_list": "HS_Right_GES_train_list.txt"
        },
        "HS_Right_GES_test": {
            "data_dir": "HS_Right_GES",
            "data_list": "HS_Right_GES_test_list.txt"
        },
        "HS_Middle_GES_train": {
            "data_dir": "HS_Right_GES",
            "data_list": "HS_Right_GES_train_list.txt"
        },
        "HS_Middle_GES_test": {
            "data_dir": "HS_Middle_GES",
            "data_list": "HS_Middle_GES_test_list.txt"
        },
        "HS_Left_GES_train": {
            "data_dir": "HS_Left_GES",
            "data_list": "HS_Left_GES_train_list.txt"
        },
        "HS_Left_GES_test": {
            "data_dir": "HS_Left_GES",
            "data_list": "HS_Left_GES_test_list.txt"
        },
        "HY_GES_train": {
            "data_dir": "HY_GES",
            "data_list": "HY_GES_train_list.txt"
        },
        "HY_GES_test": {
            "data_dir": "HY_GES",
            "data_list": "HY_GES_test_list.txt"
        },
        "DXH_AS_GES_train": {
            "data_dir": "DXH_AS_GES",
            "data_list": "DXH_AS_GES_train_list.txt"
        },
        "DXH_AS_GES_test": {
            "data_dir": "DXH_AS_GES",
            "data_list": "DXH_AS_GES_test_list.txt"
        },
        "DXH_IMG_train": {
            "data_dir": "DXH_IMG",
            "data_list": "DXH_IMG_train_list.txt"
        },
        "DXH_GES_test_paper": {
            "data_dir": "DXH",
            "data_list": "DXH_IMG_test_list.txt"
        },
        "HN_AS_GES_train": {
            "data_dir": "HN_AS_GES",
            "data_list": "HN_AS_GES_train_list.txt"
        },
        "HN_AS_GES_test": {
            "data_dir": "HN_AS_GES",
            "data_list": "HN_AS_GES_test_list.txt"
        },
        "XZ_AS_GES_train": {
            "data_dir": "XZ_AS_GES",
            "data_list": "XZ_AS_GES_train_list.txt"
        },
        "XZ_AS_GES_test": {
            "data_dir": "XZ_AS_GES",
            "data_list": "XZ_AS_GES_test_list.txt"
        },
        "CD_AS_GES_train": {
            "data_dir": "CD_AS_GES",
            "data_list": "CD_AS_GES_train_list.txt"
        },
        "CD_AS_GES_test": {
            "data_dir": "CD_AS_GES",
            "data_list": "CD_AS_GES_test_list.txt"
        },
        "HP_AS_GES_train": {
            "data_dir": "HP_AS_GES",
            "data_list": "HP_AS_GES_train_list.txt"
        },
        "HP_AS_GES_test": {
            "data_dir": "HP_AS_GES",
            "data_list": "HP_AS_GES_test_list.txt"
        },
        "JX_AS_GES_train": {
            "data_dir": "JX_AS_GES",
            "data_list": "JX_AS_GES_train_list.txt"
        },
        "JX_AS_GES_test": {
            "data_dir": "JX_AS_GES",
            "data_list": "JX_AS_GES_test_list.txt"
        },
        "WC_GES_train": {
            "data_dir": "WC_GES",
            "data_list": "WC_GES_train_list.txt"
        },
        "WC_GES_test": {
            "data_dir": "WC_GES",
            "data_list": "WC_GES_test_list.txt"
        },
        "WC_IMG_train": {
            "data_dir": "WC_IMG",
            "data_list": "WC_IMG_train_list.txt"
        },
        "WC_IMG_test": {
            "data_dir": "WC_IMG",
            "data_list": "WC_IMG_test_list.txt"
        },
        "WC_GES_test_paper": {
            "data_dir": "WC",
            "data_list": "WC_IMG_test_list.txt"
        },
        "JH_GES_train": {
            "data_dir": "JH_GES",
            "data_list": "JH_GES_train_list.txt"
        },
        "JH_GES_test": {
            "data_dir": "JH_GES",
            "data_list": "JH_GES_test_list.txt"
        },
        "JH_GES_test_paper": {
            "data_dir": "JH",
            "data_list": "JH_IMG_test_list.txt"
        },
        "QS_GES_train": {
            "data_dir": "QS_GES",
            "data_list": "QS_GES_train_list.txt"
        },
        "QS_GES_test": {
            "data_dir": "QS_GES",
            "data_list": "QS_GES_test_list.txt"
        },
        "QK_GES_train": {
            "data_dir": "QK_GES",
            "data_list": "QK_GES_train_list.txt"
        },
        "QK_GES_test": {
            "data_dir": "QK_GES",
            "data_list": "QK_GES_test_list.txt"
        },
        "QK_TDT_train": {
            "data_dir": "QK_TDT",
            "data_list": "QK_TDT_train_list.txt"
        },
        "QK_TDT_test": {
            "data_dir": "QK_TDT",
            "data_list": "QK_TDT_test_list.txt"
        },
        "wh_wv2_train": {
            "data_dir": "wh_wv2",
            "data_list": "wh_wv2_train_list.txt"
        },
        "NJ_HK_train": {
            "data_dir": "NJ_HK/images",
            "data_list": "NJ_HK_train_list.txt"
        },
        "NJ_HK_test": {
            "data_dir": "NJ_HK/images",
            "data_list": "NJ_HK_test_list.txt"
        },
        "NJ_WX_train": {
            "data_dir": "NJ_WX/images",
            "data_list": "NJ_WX_train_list.txt"
        },
        "NJ_WX_test": {
            "data_dir": "NJ_WX/images",
            "data_list": "NJ_WX_test_list.txt"
        },
        "vegas_wv2_train": {
            "data_dir": "vegas_wv2",
            "data_list": "vegas_wv2_train_list.txt"
        },
        "vegas_wv2_test": {
            "data_dir": "vegas_wv2",
            "data_list": "vegas_wv2_test_list.txt"
        },
        "PotsdamRGB_train": {
            "data_dir": "PotsdamRGB",
            "data_list": "PotsdamRGB_all.txt"
        },
        "PotsdamRGB_960_train": {
            "data_dir": "PotsdamRGB_960",
            "data_list": "PotsdamRGB_960_all.txt"
        },
        "PotsdamIRGB_train": {
            "data_dir": "PotsdamIRGB",
            "data_list": "PotsdamIRGB_all.txt"
        },
        "PotsdamIRGB_960_train": {
            "data_dir": "PotsdamIRGB_960",
            "data_list": "PotsdamIRGB_960_all.txt"
        },
        "Vaihingen_train": {
            "data_dir": "Vaihingen",
            "data_list": "Vaihingen_all.txt"
        },
        "Vaihingen_test": {
            "data_dir": "Vaihingen",
            "data_list": "Vaihingen_test.txt"
        },
        "Vaihingen_self_distill_train": {
            "data_dir": "Vaihingen",
            "data_list": "Vaihingen_all.txt",
            "label_dir": "Vaihingen/soft_labels/inference/Vaihingen_train"
        },
    }

    @staticmethod
    def get(name, mode, num_classes, max_iters=None, transform=None):
        if "gta5" in name:
            data_dir = DatasetCatalog.DATASET_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["data_dir"]),
                data_list=os.path.join(data_dir, attrs["data_list"]),
            )
            return GTA5DataSet(args["root"], args["data_list"], max_iters=max_iters, num_classes=num_classes,
                               split=mode, transform=transform)
        elif "cityscapes" in name:
            data_dir = DatasetCatalog.DATASET_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["data_dir"]),
                data_list=os.path.join(data_dir, attrs["data_list"]),
            )
            if 'distill' in name:
                args['label_dir'] = os.path.join(data_dir, attrs["label_dir"])
                return cityscapesSelfDistillDataSet(args["root"], args["data_list"], args['label_dir'],
                                                    max_iters=max_iters, num_classes=num_classes, split=mode,
                                                    transform=transform)
            return cityscapesDataSet(args["root"], args["data_list"], max_iters=max_iters, num_classes=num_classes,
                                     split=mode, transform=transform)
        elif "sh_wv2" in name:
            data_dir = DatasetCatalog.RS_DATASET_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["data_dir"]),
                data_list=os.path.join(data_dir, attrs["data_list"]),
            )
            return sh_wv2_Dataset(args["root"], args["data_list"], max_iters=max_iters, num_classes=num_classes,
                                  split=mode, transform=transform)
        elif "JA_GES" in name:
            data_dir = DatasetCatalog.RS_DATASET_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["data_dir"]),
                data_list=os.path.join(data_dir, attrs["data_list"]),
            )
            if 'distill' in name:
                args['label_dir'] = os.path.join(data_dir, attrs["label_dir"])
                return JAGESSelfDistillDataSet(args["root"], args["data_list"], args['label_dir'],
                                                   max_iters=max_iters, num_classes=num_classes, split=mode,
                                                   transform=transform)
            return JA_GES_Dataset(args["root"], args["data_list"], max_iters=max_iters, num_classes=num_classes,
                                     split=mode, transform=transform)
        elif "HS_GES" in name:
            data_dir = DatasetCatalog.RS_DATASET_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["data_dir"]),
                data_list=os.path.join(data_dir, attrs["data_list"]),
            )
            return HS_GES_Dataset(args["root"], args["data_list"], max_iters=max_iters, num_classes=num_classes,
                                     split=mode, transform=transform)
        elif "HS_Right_GES" in name:
            data_dir = DatasetCatalog.RS_DATASET_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["data_dir"]),
                data_list=os.path.join(data_dir, attrs["data_list"]),
            )
            return HS_Right_GES_Dataset(args["root"], args["data_list"], max_iters=max_iters, num_classes=num_classes,
                                     split=mode, transform=transform)
        elif "HS_Middle_GES" in name:
            data_dir = DatasetCatalog.RS_DATASET_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["data_dir"]),
                data_list=os.path.join(data_dir, attrs["data_list"]),
            )
            return HS_Middle_GES_Dataset(args["root"], args["data_list"], max_iters=max_iters, num_classes=num_classes,
                                     split=mode, transform=transform)
        elif "HS_Left_GES" in name:
            data_dir = DatasetCatalog.RS_DATASET_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["data_dir"]),
                data_list=os.path.join(data_dir, attrs["data_list"]),
            )
            return HS_Left_GES_Dataset(args["root"], args["data_list"], max_iters=max_iters, num_classes=num_classes,
                                     split=mode, transform=transform)
        elif "HY_GES" in name:
            data_dir = DatasetCatalog.RS_DATASET_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["data_dir"]),
                data_list=os.path.join(data_dir, attrs["data_list"]),
            )
            return HY_GES_Dataset(args["root"], args["data_list"], max_iters=max_iters, num_classes=num_classes,
                                     split=mode, transform=transform)
        elif "DXH_AS_GES" in name:
            data_dir = DatasetCatalog.RS_DATASET_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["data_dir"]),
                data_list=os.path.join(data_dir, attrs["data_list"]),
            )
            return DXH_AS_GES_Dataset(args["root"], args["data_list"], max_iters=max_iters, num_classes=num_classes,
                                     split=mode, transform=transform)
        elif "DXH_IMG" in name:
            data_dir = DatasetCatalog.RS_DATASET_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["data_dir"]),
                data_list=os.path.join(data_dir, attrs["data_list"]),
            )
            return DXH_AS_GES_Dataset(args["root"], args["data_list"], max_iters=max_iters, num_classes=num_classes,
                                     split=mode, transform=transform)
        elif "DXH_GES_test_paper" in name:
            data_dir = DatasetCatalog.WH_GES_TEST_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["data_dir"]),
                data_list=os.path.join(data_dir, attrs["data_list"]),
            )
            return DXH_AS_GES_Dataset(args["root"], args["data_list"], max_iters=max_iters, num_classes=num_classes,
                                     split=mode, transform=transform)
        elif "HN_AS_GES" in name:
            data_dir = DatasetCatalog.RS_DATASET_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["data_dir"]),
                data_list=os.path.join(data_dir, attrs["data_list"]),
            )
            return HN_AS_GES_Dataset(args["root"], args["data_list"], max_iters=max_iters, num_classes=num_classes,
                                     split=mode, transform=transform)
        elif "XZ_AS_GES" in name:
            data_dir = DatasetCatalog.RS_DATASET_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["data_dir"]),
                data_list=os.path.join(data_dir, attrs["data_list"]),
            )
            return XZ_AS_GES_Dataset(args["root"], args["data_list"], max_iters=max_iters, num_classes=num_classes,
                                     split=mode, transform=transform)
        elif "CD_AS_GES" in name:
            data_dir = DatasetCatalog.RS_DATASET_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["data_dir"]),
                data_list=os.path.join(data_dir, attrs["data_list"]),
            )
            return CD_AS_GES_Dataset(args["root"], args["data_list"], max_iters=max_iters, num_classes=num_classes,
                                     split=mode, transform=transform)
        elif "HP_AS_GES" in name:
            data_dir = DatasetCatalog.RS_DATASET_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["data_dir"]),
                data_list=os.path.join(data_dir, attrs["data_list"]),
            )
            return HP_AS_GES_Dataset(args["root"], args["data_list"], max_iters=max_iters, num_classes=num_classes,
                                     split=mode, transform=transform)
        elif "JX_AS_GES" in name:
            data_dir = DatasetCatalog.RS_DATASET_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["data_dir"]),
                data_list=os.path.join(data_dir, attrs["data_list"]),
            )
            return JX_AS_GES_Dataset(args["root"], args["data_list"], max_iters=max_iters, num_classes=num_classes,
                                     split=mode, transform=transform)
        elif "WC_GES_train" == name:
            data_dir = DatasetCatalog.RS_DATASET_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["data_dir"]),
                data_list=os.path.join(data_dir, attrs["data_list"]),
            )
            return WC_GES_Dataset(args["root"], args["data_list"], max_iters=max_iters, num_classes=num_classes,
                                     split=mode, transform=transform)
        elif "WC_IMG" in name:
            data_dir = DatasetCatalog.RS_DATASET_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["data_dir"]),
                data_list=os.path.join(data_dir, attrs["data_list"]),
            )
            return WC_GES_Dataset(args["root"], args["data_list"], max_iters=max_iters, num_classes=num_classes,
                                     split=mode, transform=transform)
        elif "WC_GES_test_paper" == name:
            data_dir = DatasetCatalog.WH_GES_TEST_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["data_dir"]),
                data_list=os.path.join(data_dir, attrs["data_list"]),
            )
            return WC_GES_Dataset(args["root"], args["data_list"], max_iters=max_iters, num_classes=num_classes,
                                     split=mode, transform=transform)
        elif "JH_GES_train" == name:
            data_dir = DatasetCatalog.RS_DATASET_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["data_dir"]),
                data_list=os.path.join(data_dir, attrs["data_list"]),
            )
            return JH_GES_Dataset(args["root"], args["data_list"], max_iters=max_iters, num_classes=num_classes,
                                     split=mode, transform=transform)
        elif "JH_GES_test_paper" == name:
            data_dir = DatasetCatalog.WH_GES_TEST_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["data_dir"]),
                data_list=os.path.join(data_dir, attrs["data_list"]),
            )
            return JH_GES_Dataset(args["root"], args["data_list"], max_iters=max_iters, num_classes=num_classes,
                                     split=mode, transform=transform)
        elif "QS_GES" in name:
            data_dir = DatasetCatalog.RS_DATASET_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["data_dir"]),
                data_list=os.path.join(data_dir, attrs["data_list"]),
            )
            return QS_GES_Dataset(args["root"], args["data_list"], max_iters=max_iters, num_classes=num_classes,
                                     split=mode, transform=transform)
        elif "QK_GES" in name:
            data_dir = DatasetCatalog.RS_DATASET_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["data_dir"]),
                data_list=os.path.join(data_dir, attrs["data_list"]),
            )
            return QK_GES_Dataset(args["root"], args["data_linst"], max_iters=max_iters, num_classes=num_classes,
                                     split=mode, transform=transform)
        elif "QK_TDT" in name:
            data_dir = DatasetCatalog.RS_DATASET_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["data_dir"]),
                data_list=os.path.join(data_dir, attrs["data_list"]),
            )
            return QK_TDT_Dataset(args["root"], args["data_list"], max_iters=max_iters, num_classes=num_classes,
                                     split=mode, transform=transform)
        elif "wh_wv2" in name:
            data_dir = DatasetCatalog.RS_DATASET_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["data_dir"]),
                data_list=os.path.join(data_dir, attrs["data_list"]),
            )
            return wh_wv2_Dataset(args["root"], args["data_list"], max_iters=max_iters, num_classes=num_classes,
                                  split=mode, transform=transform)
        elif "PotsdamRGB" in name:
            data_dir = DatasetCatalog.ISPRS_DATASET_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["data_dir"]),
                data_list=os.path.join(data_dir, attrs["data_list"]),
            )
            return potsdamRGBDataSet(args["root"], args["data_list"], max_iters=max_iters, num_classes=num_classes,
                                     split=mode, transform=transform)
        elif "PotsdamIRGB" in name:
            data_dir = DatasetCatalog.ISPRS_DATASET_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["data_dir"]),
                data_list=os.path.join(data_dir, attrs["data_list"]),
            )
            return potsdamIRGBDataSet(args["root"], args["data_list"], max_iters=max_iters, num_classes=num_classes,
                                      split=mode, transform=transform)
        elif "PotsdamRGB_960" in name:
            data_dir = DatasetCatalog.ISPRS_DATASET_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["data_dir"]),
                data_list=os.path.join(data_dir, attrs["data_list"]),
            )
            return potsdamRGBDataSet(args["root"], args["data_list"], max_iters=max_iters, num_classes=num_classes,
                                     split=mode, transform=transform)
        elif "PotsdamIRGB_960" in name:
            data_dir = DatasetCatalog.ISPRS_DATASET_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["data_dir"]),
                data_list=os.path.join(data_dir, attrs["data_list"]),
            )
            return potsdamIRGBDataSet(args["root"], args["data_list"], max_iters=max_iters, num_classes=num_classes,
                                      split=mode, transform=transform)
        elif "Vaihingen" in name:
            data_dir = DatasetCatalog.ISPRS_DATASET_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["data_dir"]),
                data_list=os.path.join(data_dir, attrs["data_list"]),
            )
            if 'distill' in name:
                args['label_dir'] = os.path.join(data_dir, attrs["label_dir"])
                return vaihingenSelfDistillDataSet(args["root"], args["data_list"], args['label_dir'],
                                                   max_iters=max_iters, num_classes=num_classes, split=mode,
                                                   transform=transform)
            return vaihingenDataSet(args["root"], args["data_list"], max_iters=max_iters, num_classes=num_classes,
                                    split=mode, transform=transform)

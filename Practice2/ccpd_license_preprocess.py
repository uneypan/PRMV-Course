import patoolib
from datasets.plateLabel import plateLabel
import os
if not os.path.exists("../CCPD_LISENCE/val_verify"):
    patoolib.extract_archive("../CCPD_LISENCE/val_verify.rar", outdir="../CCPD_LISENCE")
if not os.path.exists("../CCPD_LISENCE/CCPD_CRPD_OTHER_ALL"):
    patoolib.extract_archive("../CCPD_LISENCE/train_plate.tar.gz", outdir="../CCPD_LISENCE")
plateLabel("../CCPD_LISENCE/CCPD_CRPD_OTHER_ALL","datasets/train.txt")
plateLabel("../CCPD_LISENCE/val_verify","datasets/val.txt")
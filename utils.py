from enum import Enum


class Split(str, Enum):
    train = "train"
    test = "test"
    dev = "dev"


class Collection(str, Enum):
    tib_core_subjects = "tib-core-subjects"
    all_subjects = "all-subjects"


class Prefix(str, Enum):
    bi_encoder = "bi-encoder"
    cross_encoder = "cross-encoder"

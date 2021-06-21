from collections import namedtuple
from enum import Enum, unique

MetricType = namedtuple("Metirc", ["name", "accumulation"])


@unique
class ModelState(Enum):
    Train = 777
    Evaluate = 7777
    Test = 77777

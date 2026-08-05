from enum import Enum


class AlgorithmChoice(Enum):
    Unknown = "Unknown"
    DES = "DES"
    MFCMAES = "MFCMAES"
    CMAES = "CMAES"
    LBFGSB = "LBFGSB"
    POWELL = "POWELL"
    NELDERMEAD = "NELDERMEAD"
    BFGS = "BFGS"

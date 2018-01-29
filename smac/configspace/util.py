from typing import List

import numpy as np

from smac.configspace import Configuration, ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformFloatHyperparameter, UniformIntegerHyperparameter, Constant, \
    OrdinalHyperparameter


def convert_configurations_to_array(
        configs: List[Configuration]
) -> np.ndarray:
    """Impute inactive hyperparameters in configurations with their default.

    Necessary to apply an EPM to the data.

    Parameters
    ----------
    configs : List[Configuration]
        List of configuration objects.

    Returns
    -------
    np.ndarray
        Array with configuration hyperparameters. Inactive values are imputed
        with their default value.
    """
    configs_array = np.array([config.get_array() for config in configs],
                             dtype=np.float64)
    configuration_space = configs[0].configuration_space
    return impute_default_values(configuration_space, configs_array)


def impute_default_values(
        configuration_space: ConfigurationSpace,
        configs_array: np.ndarray
) -> np.ndarray:
    """Impute inactive hyperparameters in configuration array with -1.

    Necessary to apply an EPM to the data.

    Parameters
    ----------
    configuration_space : ConfigurationSpace

    configs_array : np.ndarray
        Array of configurations.

    Returns
    -------
    np.ndarray
        Array with configuration hyperparameters. Inactive values are imputed
        with their default value.
    """

    for idx, hp in enumerate(configuration_space.get_hyperparameters()):
        parents = configuration_space.get_parents_of(hp.name)
        if len(parents) == 0:
            continue
        else:
            if isinstance(hp, CategoricalHyperparameter):
                impute_values = len(hp.choices)
            elif isinstance(hp, (UniformFloatHyperparameter,
                                 UniformIntegerHyperparameter)):
                impute_values = -1
            elif isinstance(hp, Constant):
                impute_values = 1
            else:
                raise ValueError

        nonfinite_mask = ~np.isfinite(configs_array[:, idx])
        configs_array[nonfinite_mask, idx] = impute_values
    return configs_array
from typing import Any, Dict, Type


class Config:
    """Config needed to instanciate a MockerFixture

    Example:
    MockerFixture(Config())
    """

    def getini(self, *args):
        return False


def modify_default_values(
    class_method: Type[Any],
    new_param_defaults: Dict[Any, Any] = {},
) -> None:
    """modify default values of a method

    Parameters
    ----------
    class_method : Type[Any]
        method obj of a class (e.g. Class.Method)
    new_param_defaults : Dict[Any, Any], optional
        new default parameter values, by default {}
    """

    for param_name, param_value in new_param_defaults.items():
        if class_method.__defaults__:
            chunk_size_id = list(class_method.__annotations__.keys()).index(
                param_name
            )
            init_defaults_values = list(class_method.__defaults__)
            init_defaults_values[chunk_size_id] = param_value
            class_method.__defaults__ = tuple(init_defaults_values)
        else:
            class_method.__kwdefaults__[param_name] = param_value

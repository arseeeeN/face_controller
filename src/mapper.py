from enum import Enum
from collections.abc import Callable
from actions import Action


class ParamName(Enum):
    BODY_ANGLE_X = 1
    BODY_ANGLE_Y = 2
    BODY_ANGLE_Z = 3
    ANGLE_X = 4
    ANGLE_Y = 5
    ANGLE_Z = 6
    MOUTH_X = 7
    MOUTH_OPEN_Y = 8
    MOUTH_FORM = 9
    BROW_L_Y = 10
    BROW_R_Y = 11
    EYE_BALL_X = 12
    EYE_BALL_Y = 13
    EYE_R_OPEN = 14
    EYE_L_OPEN = 15


class Parameter:
    name: ParamName
    value: float

    def __init__(self, name, value):
        self.name = name
        self.value = value


class ParameterTransformer:
    transformer: Callable[[list[Parameter]], float]
    parameter_references: list[Parameter]

    def __init__(self, transformer, parameters=[]):
        self.parameter_references = parameters
        self.transformer = transformer

    def get_action_value(self) -> float:
        return self.transformer(self.parameter_references)


def average(parameters: list[Parameter]) -> float:
    res = 0.0
    for param in parameters:
        res += param.value
    return res / len(parameters)


def single(parameters: list[Parameter]) -> float:
    return parameters[0].value


def wrap_threshold(transformer, threshold: float, above: float, below: float) -> Callable[[list[Parameter]], float]:
    return lambda x: above if transformer(x) >= threshold else below


def wrap_piecewise(transformer, max: float, min: float, above: float, below: float, inside=0.0) -> Callable[[list[Parameter]], float]:
    def piecewise(x):
        x = transformer(x)
        if x >= max:
            return above
        elif x <= min:
            return below
        else:
            return inside
    return piecewise


class ActionParameterMapper:
    map: dict[Action, ParameterTransformer]
    parameters: dict[ParamName, Parameter]

    def __init__(self):
        self.map = {}
        self.parameters = {}

    def get_action_value(self, action: Action) -> float:
        return self.map[action].get_action_value()

    def create_mapping(self, action: Action, parameter_transformer: ParameterTransformer):
        self.map[action] = parameter_transformer
        for param in parameter_transformer.parameter_references:
            self.set_parameter(param, action)

    def create_empty_mapping(self, action: Action, transformer: Callable[[list[Parameter]], float]):
        self.map[action] = ParameterTransformer(transformer, [])

    def set_parameter(self, parameter: Parameter, action: Action):
        if action in self.map:
            if parameter.name in self.parameters:
                parameter = self.parameters[parameter.name]
                param_references = self.map[action].parameter_references
                for i, param in enumerate(param_references):
                    if param.name == parameter.name:
                        param_references[i] = parameter
                        break
            else:
                self.parameters[parameter.name] = parameter

    def add_parameter(self, name: ParamName, action: Action):
        if action in self.map:
            if name in self.parameters:
                parameter = self.parameters[name]
                self.map[action].parameter_references.append(parameter)
            else:
                parameter = Parameter(name, 0.0)
                self.map[action].parameter_references.append(parameter)
                self.parameters[name] = parameter

    def set_parameter_value(self, name: ParamName, value: float):
        if name in self.parameters:
            self.parameters[name].value = value

    def trigger_actions(self):
        for action, transformer in self.map.items():
            action.trigger(transformer.get_action_value())

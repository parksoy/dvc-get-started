from ruamel.yaml import YAML
from pydantic import BaseModel, Field
from typing import Literal


class PrepareParams(BaseModel):
    seed: int
    remix: bool
    remix_split: float = Field(ge=0.0, le=1.0)


class PreprocessParams(BaseModel):
    seed: int
    normalize: bool
    shuffle: bool
    add_noise: bool
    noise_amount: float
    noise_s_vs_p: float


class TrainParams(BaseModel):
    seed: int
    validation_split: float = Field(ge=0.0, le=1.0)
    epochs: int = Field(gt=0)
    batch_size: int = Field(gt=0)


class MlpParams(BaseModel):
    units: int
    activation: str


class CnnParams(BaseModel):
    dense_units: int
    activation: str
    conv_kernel_size: int
    conv_units: int
    dropout: float = Field(ge=0.0, le=1.0)


class ModelParams(BaseModel):
    name: Literal["mlp", "cnn"]
    optimizer: str

def load_params():
    yaml = YAML(typ="safe")
    with open("params.yaml") as f:
        params = yaml.load(f)
    return params


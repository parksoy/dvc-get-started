# Migration TODO

## Tasks (complete in order, one per run)

- [x] requirements.txt — add `pydantic>=2.0` on a new line. Skip if already present.

- [x] src/util.py part 1 — add six Pydantic section models. Add these imports and classes, do not touch load_params() yet:
  ```python
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
  ```

- [x] src/util.py part 2 — add root Params model and update load_params() to return it:
  ```python
  class Params(BaseModel):
      prepare: PrepareParams
      preprocess: PreprocessParams
      train: TrainParams
      model: ModelParams
      mlp: MlpParams
      cnn: CnnParams

  def load_params() -> Params:
      yaml = YAML(typ="safe")
      with open("params.yaml") as f:
          raw = yaml.load(f)
      return Params(**raw)
  ```

- [x] src/prepare.py — replace dict access with attribute access. Pattern:
  Before: `params = load_params()["prepare"]` / `params["seed"]`
  After:  `params = load_params().prepare` / `params.seed`

- [x] src/preprocess.py — same pattern, section is `preprocess`.

- [x] src/train.py — same pattern, section is `train`.

- [ ] src/models.py — same pattern, sections are `model`, `mlp`, `cnn`. Nested example:
  Before: `load_params()["mlp"]["units"]`
  After:  `load_params().mlp.units`

- [ ] src/evaluate.py — same pattern:
  Before: `load_params()["train"]["batch_size"]`
  After:  `load_params().train.batch_size`

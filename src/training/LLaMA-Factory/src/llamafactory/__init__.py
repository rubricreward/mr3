# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""Efficient fine-tuning of large language models.

Level:
  api, webui > chat, eval, train > data, model > hparams > extras

Disable version checking: DISABLE_VERSION_CHECK=1
Enable VRAM recording: RECORD_VRAM=1
Force using torchrun: FORCE_TORCHRUN=1
Set logging verbosity: LLAMAFACTORY_VERBOSITY=WARN
Use modelscope: USE_MODELSCOPE_HUB=1
Use openmind: USE_OPENMIND_HUB=1
"""

from .extras.env import VERSION
import torch.serialization
from deepspeed.runtime.zero.config import ZeroStageEnum
from deepspeed.runtime.fp16.loss_scaler import LossScaler
from deepspeed.runtime.zero.stage_1_and_2 import DeepSpeedZeroOptimizer
from deepspeed.runtime.zero.stage3 import DeepSpeedZeroOptimizer_Stage3

torch.serialization.add_safe_globals([
    ZeroStageEnum,
    LossScaler,
    DeepSpeedZeroOptimizer,
    DeepSpeedZeroOptimizer_Stage3
])

__version__ = VERSION

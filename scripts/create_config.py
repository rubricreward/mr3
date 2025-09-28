import os
import logging
import argparse
import os
import json

import socket
import json
from abc import ABC, abstractmethod
from typing import Dict, Any

import torch

CUR_DIR = os.path.abspath(os.path.dirname(__file__))
ROOT_DIR = os.path.abspath(os.path.dirname(CUR_DIR))
MAX_INPUT_TOKENS = 16384
DEFAULT_OUTPUT_TOKENS = 16384

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

class ModelConfigBuilder(ABC):
    """Abstract base class for model configuration builders."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset the builder to start fresh."""
        self.config = {
            "model_name": "",
            "provider_name": "local",
            "model_args": {},
            "generation_args": {}
        }
    
    @abstractmethod
    def set_model_defaults(self) -> 'ModelConfigBuilder':
        """Set model-specific defaults."""
        pass
    
    def set_model_name(self, model_name) -> 'ModelConfigBuilder':
        self.config['model_name'] = model_name
        return self
    
    def set_num_gpus(self, num_gpus) -> 'ModelConfigBuilder':
        self.config['model_args']['tensor_parallel_size'] = num_gpus
        return self
    
    def set_model_args(self, max_num_seqs, max_input_tokens, max_output_tokens) -> 'ModelConfigBuilder':
        # TODO: Make it more flexible...
        self.config['model_args']['max_num_seqs'] = max_num_seqs
        self.config['generation_args']['max_tokens'] = max_output_tokens
        self.config['model_args']['max_model_len'] = max_input_tokens + max_output_tokens
        return self
    
    def _find_free_port(self, start_port=8080, end_port=9000):
        """Find a free port in the specified range."""
        for port in range(start_port, end_port):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                try:
                    sock.bind(('localhost', port))
                    return port
                except OSError:
                    continue
        raise RuntimeError(f"No free ports available in range {start_port}-{end_port}")
        
    def set_provider(self, provider_name) -> 'ModelConfigBuilder':
        """Configure for VLLM provider."""
        if provider_name == "openai":
            self.config["provider_name"] = "openai"
            port = self._find_free_port(start_port=8080, end_port=9000)
            self.config["api_base_url"] = f"http://localhost:{port}/v1"
            self.config["num_workers"] = self.config['model_args'].get('max_num_seqs', 256)
            self.config["model_args"]["enable_reasoning"] = True
            self.config["model_args"]["reasoning_parser"] = "deepseek_r1"
            if "top_k" in self.config["generation_args"]:
                top_k = self.config["generation_args"].pop("top_k")
                self.config["generation_args"]["extra_body"] = {"top_k": top_k}

        return self
    
    def set_surgery_mode(self, enabled: bool = True) -> 'ModelConfigBuilder':
        """Enable surgery mode (typically requires more tokens)."""
        if enabled:
            # Increase token limits for surgery
            self.config["generation_args"]["max_tokens"] = self.config["generation_args"].get("max_tokens", DEFAULT_OUTPUT_TOKENS) * 2
            self.config["model_args"]["max_model_len"] = self.config["generation_args"]["max_tokens"] + MAX_INPUT_TOKENS
            
            if "num_workers" in self.config:
                self.config['num_workers'] = self.config['model_args']['max_num_seqs']

        return self

    def build(self) -> Dict[str, Any]:
        """Build and return the final configuration."""
        return self.config.copy()


class Llama32ConfigBuilder(ModelConfigBuilder):
    """Builder for Llama-3.2 model configurations."""

    def set_model_defaults(self) -> 'ModelConfigBuilder':
        """Set Llama-3.2-specific defaults."""
        self.config.update({
            "model_name": "meta-llama/Llama-3.2-3B-Instruct",
            "provider_name": "local",
            "enable_thinking": False,
            "model_args": {
                "tensor_parallel_size": 1,
                "max_num_seqs": 256,
                "max_model_len": DEFAULT_OUTPUT_TOKENS + MAX_INPUT_TOKENS,
                "gpu_memory_utilization": 0.95,
                "dtype": "bfloat16",
                "enforce_eager": True,
            },
            "generation_args": {
                "temperature": 0.6,
                "top_p": 0.9,
                "max_tokens": DEFAULT_OUTPUT_TOKENS,
                "top_k": 20
            }
        })
        return self


class Qwen3ConfigBuilder(ModelConfigBuilder):
    """Builder for Qwen3 model configurations."""
    
    def set_model_defaults(self) -> 'ModelConfigBuilder':
        """Set Qwen3-specific defaults."""
        self.config.update({
            "model_name": "Qwen/Qwen3-4B",
            "provider_name": "local",
            "enable_thinking": True,
            "model_args": {
                "tensor_parallel_size": 1,
                "max_num_seqs": 256,
                "max_model_len": DEFAULT_OUTPUT_TOKENS + MAX_INPUT_TOKENS,
                "gpu_memory_utilization": 0.95,
                "dtype": "bfloat16",
                "enforce_eager": True,
            },
            "generation_args": {
                "temperature": 0.6,
                "top_p": 0.95,
                "max_tokens": DEFAULT_OUTPUT_TOKENS,
                "top_k": 20
            }
        })
        return self
    
class GptOssConfigBuilder(ModelConfigBuilder):
    """Builder for gpt-oss model configurations."""
    
    def set_model_defaults(self) -> 'ModelConfigBuilder':
        """Set gpt-oss-specific defaults."""
        self.config.update({
            "model_name": "openai/gpt-oss-20b",
            "provider_name": "local",
            "model_args": {
                "tensor_parallel_size": 1,
                "max_num_seqs": 512,
                "max_model_len": DEFAULT_OUTPUT_TOKENS + MAX_INPUT_TOKENS,
                "gpu_memory_utilization": 0.95,
                "dtype": "bfloat16",
                "trust_remote_code": True,
                "enforce_eager": True
            },
            "generation_args": {
                "stop": ["<|return|>", "<|call|>"],
                "max_tokens": DEFAULT_OUTPUT_TOKENS,
            }
        })
        return self

class QwenConfigBuilder(ModelConfigBuilder):
    """Builder for Qwen model configurations."""
    
    def set_model_defaults(self) -> 'ModelConfigBuilder':
        """Set Qwen specific defaults."""
        self.config.update({
            "model_name": "Qwen/Qwen2.5-7B-Instruct",
            "provider_name": "local",
            "enable_thinking": None,
            "model_args": {
                "tensor_parallel_size": 1,
                "max_num_seqs": 256,
                "max_model_len": DEFAULT_OUTPUT_TOKENS + MAX_INPUT_TOKENS,
                "gpu_memory_utilization": 0.95,
                "dtype": "bfloat16",
                "enforce_eager": True,
            },
            "generation_args": {
                "temperature": 0.6,
                "top_p": 0.95,
                "max_tokens": DEFAULT_OUTPUT_TOKENS,
                "top_k": 20
            }
        })
        return self

class Phi4ConfigBuilder(ModelConfigBuilder):
    """Builder for Phi-4 model configurations."""
    
    def set_model_defaults(self) -> 'ModelConfigBuilder':
        """Set Phi-4-specific defaults."""
        self.config.update({
            "model_name": "microsoft/Phi-4-reasoning-plus",
            "provider_name": "local",
            "enable_thinking": None,
            "model_args": {
                "tensor_parallel_size": 1,
                "max_num_seqs": 256,
                "max_model_len": DEFAULT_OUTPUT_TOKENS + MAX_INPUT_TOKENS,
                "gpu_memory_utilization": 0.95,
                "dtype": "bfloat16",
                "enforce_eager": True,
            },
            "generation_args": {
                "temperature": 0.8,
                "top_p": 0.95,
                "max_tokens": DEFAULT_OUTPUT_TOKENS,
                "top_k": 50,
            }
        })
        return self
    
class LLaMAConfigBuilder(ModelConfigBuilder):
    """Builder for LLaMA model configurations."""
    
    def set_model_defaults(self) -> 'ModelConfigBuilder':
        """Set LLaMA -specific defaults."""
        self.config.update({
            "model_name": "meta-llama/Llama-3.2-3B-Instruct",
            "provider_name": "local",
            "model_args": {
                "tensor_parallel_size": 1,
                "max_num_seqs": 128,
                "max_model_len": DEFAULT_OUTPUT_TOKENS + MAX_INPUT_TOKENS,
                "gpu_memory_utilization": 0.95,
                "dtype": "bfloat16",
                "enforce_eager": True
            },
            "generation_args": {
                "temperature": 0.6,
                "max_tokens": DEFAULT_OUTPUT_TOKENS
            },
            "enable_thinking": None
        })
        return self

class ConfigBuilderFactory:
    """Factory to create appropriate builders."""
    
    _builders = {
        "qwen3": Qwen3ConfigBuilder,
        "qwen": QwenConfigBuilder,
        "phi4": Phi4ConfigBuilder,
        "llama": LLaMAConfigBuilder,
        "gpt-oss": GptOssConfigBuilder,
    }
    
    @classmethod
    def create_builder(cls, model_type: str) -> ModelConfigBuilder:
        """Create a builder for the specified model type."""
        model_type = model_type.lower()
        if model_type not in cls._builders:
            raise ValueError(f"Unknown model type: {model_type}. Available: {list(cls._builders.keys())}")
        
        builder = cls._builders[model_type]()
        builder.set_model_defaults()
        return builder
    
    @classmethod
    def register_builder(cls, model_type: str, builder_class: type):
        """Register a new builder type."""
        cls._builders[model_type.lower()] = builder_class


def save_config(config: Dict[str, Any], output_path: str):
    """Save configuration to JSON file."""
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)

def create_config(config_path, model_name, provider, max_num_seqs, max_input_tokens, max_output_tokens, num_gpus, surgery):
    if "qwen3" in model_name.lower():
        model_type = "qwen3"
    elif "qwen" in model_name.lower():
        model_type = "qwen"
    elif "phi-4" in model_name.lower():
        model_type = "phi4"
    elif "llama" in model_name.lower():
        model_type = "llama"
    elif "gpt-oss" in model_name.lower():
        model_type = "gpt-oss"
    else:
        raise ValueError("Unknown model... currently only supporting Qwen3 and Phi-4")
    
    builder = ConfigBuilderFactory.create_builder(model_type)
    config = (builder
              .set_model_name(model_name=model_name)
              .set_num_gpus(num_gpus=num_gpus)
              .set_model_args(max_num_seqs=max_num_seqs, max_input_tokens=max_input_tokens, max_output_tokens=max_output_tokens)
              .set_provider(provider_name=provider)
              .set_surgery_mode(enabled=surgery)
              .build())
    
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
        
    logging.info(f"Successfuly saved config to: {config_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_output_path', '-c', type=str, required=True,
                        help="Path to save model config")
    parser.add_argument('--model_name', '-m', type=str, required=True,
                        help="Model path or huggingface repository ID.")
    parser.add_argument('--provider', type=str, default='local',
                        choices=['local', 'openai'],
                        help="Provider type")
    parser.add_argument('--max_input_tokens', type=int, default=MAX_INPUT_TOKENS,
                        help="Default open tokens.")
    parser.add_argument('--max_output_tokens', type=int, default=DEFAULT_OUTPUT_TOKENS,
                        help="Default open tokens.")
    parser.add_argument('--max_num_seqs', type=int, default=2048,
                        help="Max num sequence.")
    parser.add_argument('--num_gpus', type=int, default=torch.cuda.device_count(),
                        help="Number of GPUs.")
    parser.add_argument('--surgery', action="store_true", dest="surgery",
                        help=f"Performing surgery requires more tokens.")
    parser.set_defaults(surgery=False)
    args = parser.parse_args()

    config_path = os.path.join(ROOT_DIR, args.config_output_path.strip())
    if not config_path.endswith('.json'):
        raise ValueError(f"Config file must be JSON. Got: {config_path}")
    
    create_config(config_path=config_path, model_name=args.model_name,
                  provider=args.provider, max_num_seqs=args.max_num_seqs, max_input_tokens=args.max_input_tokens, max_output_tokens=args.max_output_tokens,
                  num_gpus=args.num_gpus, surgery=args.surgery)

if __name__ == "__main__":
    main()

import subprocess
import logging
import time
import argparse
import os
import socket
import hashlib
import json
from datetime import datetime
from urllib.parse import urlparse

from openai import OpenAI

CUR_DIR = os.path.abspath(os.path.dirname(__file__))
ROOT_DIR = os.path.abspath(os.path.dirname(CUR_DIR))
LOG_DIR = os.path.join(ROOT_DIR, 'logs')
DEFAULT_PORT = 8080

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

def wait_for_server(base_url, model_name, timeout=3600):
    start_time = time.time()
    logging.info(f"Waiting for server to become ready on {base_url} ...")
    openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "sk-mock"), base_url=base_url)

    while time.time() - start_time < timeout:
        try:
            response = openai_client.chat.completions.create(
                model=model_name,
                messages=[{'role': 'user', 'content': 'Say Hi to me back please.'}],
                max_tokens=8,
            )

            logging.info("Server is up!")
            return
        except Exception:
            pass

        time.sleep(10)

    raise RuntimeError(f"Server failed to start within {timeout} seconds.")

def parse_port_from_base_url(base_url, default_port=DEFAULT_PORT):
    try:
        parsed = urlparse(base_url)
        return parsed.port or default_port
    except Exception:
        return default_port

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config', type=str, required=True,
                        help='Path to the model config JSON.')
    parser.add_argument('--new_model_config', type=str, default=None,
                        help='Path to the new model config JSON.')
    args = parser.parse_args()

    # Load model config
    config_path = os.path.join(ROOT_DIR, args.model_config.strip())
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    if not config_path.endswith('.json'):
        raise ValueError(f"Config file must be JSON. Got: {config_path}")

    with open(config_path, 'r') as f:
        config = json.load(f)

    model_name = config.get('model_name')
    if not model_name:
        raise ValueError(f"`model_name` is missing in config: {config_path}")

    base_url = config.get('api_base_url', f'http://localhost:{DEFAULT_PORT}/v1')
    port = parse_port_from_base_url(base_url)

    # Build additional args from `model_args` dictionary
    model_args = config.get('model_args', {})
    extra_args = []
    for key, value in model_args.items():
        if key == 'enforce_eager':
            # If True, pass the flag; if False, omit it
            if value == True or value.lower() == 'true':
                extra_args.append(f"--{key.replace('_', '-')}")
        elif key == 'enable_reasoning':
            if value == True or value.lower() == 'true':
                extra_args.append(f"--{key.replace('_', '-')}")
        elif isinstance(value, (str, int, float)):
            extra_args.append(f"--{key.replace('_', '-')}")
            extra_args.append(str(value))
        elif isinstance(value, dict):
            extra_args.append(f"--{key.replace('_', '-')}")
            extra_args.append(json.dumps(value))
        else:
            raise TypeError(f"Unsupported type for model_args[{key}]: {type(value)}")

    # Launch the vLLM server
    cmd = ["python3", "-m", "vllm.entrypoints.openai.api_server",
           "--model", model_name, "--disable-log-requests",
           "--port", str(port)] + extra_args

    logging.info(f"Starting vLLM server with command: {' '.join(cmd)}")

    os.makedirs(LOG_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = os.path.join(LOG_DIR, f"vllm_server_{port}_{timestamp}.log")
    log_file = open(log_file_path, "w")

    process = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)

    logging.info(f"Started vLLM server (PID={process.pid})")

    # Wait for the server to become ready
    try:
        wait_for_server(base_url=base_url, model_name=model_name)
    except Exception as e:
        logging.error(f"Server failed to start, killing process PID={process.pid}")
        process.kill()
        raise RuntimeError(e)

    # Save PID in config
    if not args.new_model_config:
        new_config_path = config_path
    else:
        new_config_path = os.path.join(ROOT_DIR, args.new_model_config.strip())
    os.makedirs(os.path.dirname(new_config_path), exist_ok=True)
    config['vllm_pid'] = process.pid
    with open(new_config_path, 'w') as f:
        json.dump(config, f, indent=2)

    logging.info(f"Server is running on port {port}. PID saved to {new_config_path}")

    # Exit while leaving the server running
    return 0

if __name__ == "__main__":
    main()

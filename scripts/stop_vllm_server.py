import os
import signal
import logging
import argparse
import os
import json

CUR_DIR = os.path.abspath(os.path.dirname(__file__))
ROOT_DIR = os.path.abspath(os.path.dirname(CUR_DIR))

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config', type=str, required=True,
                        help='Path to the model config JSON.')
    args = parser.parse_args()
    
    config_path = os.path.join(ROOT_DIR, args.model_config.strip())
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    if not config_path.endswith('.json'):
        raise ValueError(f"Config file must be JSON. Got: {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)

    pid = config.get('vllm_pid', None)
    if not pid:
        logging.warning(f"PID is not found in the config file.")
        return

    logging.info(f"Stopping server with PID {pid}")
    try:
        os.kill(pid, signal.SIGTERM)
        logging.info("SIGTERM sent.")
    except ProcessLookupError:
        logging.error("Process already exited.")
    except Exception as e:
        logging.error(f"Failed to kill process: {e}")

if __name__ == "__main__":
    main()

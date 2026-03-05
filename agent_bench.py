# agent_bench.py
import argparse
import asyncio
import yaml
import sys
import os
from typing import Dict, Any
from colorama import Fore, Back, Style
import wolframalpha
# 라이브러리 내부의 깐깐한 헤더 체크 로직을 무력화합니다.
def patched_validate_response(resp):
    # 원래는 여기서 'text/xml;charset=utf-8'이 아니면 에러를 내지만,
    # 우리는 그냥 통과시켜 버립니다.
    return True

# 실제 라이브러리 내부 함수를 우리가 만든 가짜 함수로 덮어씌웁니다.
# (버전에 따라 경로가 다를 수 있어 안전하게 예외처리)
try:
    import wolframalpha
    # 이 부분이 핵심입니다. 띄어쓰기 검증을 무시하게 만듭니다.
    wolframalpha.Client._validate_response = staticmethod(lambda resp: None)
    print(f"{Fore.GREEN}✅ WolframAlpha header check patched successfully.{Style.RESET_ALL}")
except Exception as e:
    print(f"{Fore.YELLOW}⚠️ WolframAlpha patch failed: {e}{Style.RESET_ALL}")
# --- WolframAlpha 패치 끝 ---

from run_react import main as react_main
from run_reflexion import main as reflexion_main
from run_llmcompiler import main as llmcompiler_main

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Loads the YAML configuration file from the specified path.
    """
    if not os.path.exists(config_path):
        print(f"Error: Configuration file ({config_path}) not found.", file=sys.stderr)
        sys.exit(1)
        
    with open(config_path, 'r', encoding='utf-8') as f:
        try:
            config = yaml.safe_load(f)
            return config
        except yaml.YAMLError as e:
            print(f"Error: Failed to parse YAML file: {e}", file=sys.stderr)
            sys.exit(1)
            
            
def main(args):
    config_data = load_config(args.config)
    if args.agent not in config_data["agents"]:
        print(f"Error: Section '{args.agent}' not found in '{args.config}'.", file=sys.stderr)
        sys.exit(1)
    agent_config_dict = {**config_data["global"], 
                         **config_data["agents"][args.agent]}
    agent_args = argparse.Namespace(**agent_config_dict)
    print(f"--- Running {args.agent} ---")
    print(f"Config File: {args.config}")
    print(f"Using Config: {agent_args}") 
    print("-" * 40)
    try:        
        print(f"Running agent type: {agent_args.type}")
        if agent_args.type == "react":
            react_main(agent_args)
        elif agent_args.type == "reflexion":
            reflexion_main(agent_args)
        elif agent_args.type == "llmcompiler":
            asyncio.run(llmcompiler_main(agent_args))
        else:
            print(f"Error: Unknown agent type '{agent_args.type}'", file=sys.stderr)
            sys.exit(1)
    except KeyboardInterrupt:
        print("\nExecution was interrupted by the user.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\nError: An exception occurred during agent execution: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified Agent Benchmark Runner")
    parser.add_argument(
        "--agent",
        type=str,
        help="Name of agent (must match a key in config.yaml)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the configuration file to use"
    )
    parser.add_argument("--print-log", help="Pring logs", action="store_true")
    args = parser.parse_args()
    main(args)
import time
import numpy as np
from langchain_openai import ChatOpenAI
from langgraph.errors import GraphRecursionError
from colorama import Fore, Style
from src.agents.ReAct.react import create_react_agent
from src.utils import parse_answer

from dotenv import load_dotenv
from langsmith import traceable, trace
import json
from datetime import datetime

load_dotenv()

def main(args):
    if args.host:
        host_url = f"http://{args.host}:{args.port}/v1"
    else:
        host_url = None

    score_sum = 0
    pass_count = 0
    latencies = []
    rollout_total_s = []
    rollout_llm_s = []
    rollout_tool_s = []
    rollout_tool_ratio = []
    output_dict = {}
    # per-tool aggregates (B)
    tool_calls = {}   # tool -> count
    tool_total = {}   # tool -> total seconds
    out_path = f"timing_{args.workload}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"

    def pretty_output(i):
        print(Fore.YELLOW+"=" * 30)
        print(f"Sample {i + 1}/{samples}")
        if args.workload == "webshop":
            print(f"Average score so far: {round(score_sum / (i + 1), 2)}")
        print(f"Accuracy so far: {round(pass_count / (i + 1), 2)}")
        if latencies:
            print(f"Avg. latency: {round(sum(latencies) / len(latencies), 2)} sec")
            print(f"p50 latency: {round(np.percentile(latencies, 50), 2)} sec")
            print(f"p90 latency: {round(np.percentile(latencies, 90), 2)} sec")
            print(f"p95 latency: {round(np.percentile(latencies, 95), 2)} sec") 
            print(f"p99 latency: {round(np.percentile(latencies, 99), 2)} sec")
        print("=" * 30+Style.RESET_ALL)
        print("\n")

    # Load model
    model = ChatOpenAI(model=args.model, base_url=host_url, stream_usage=True, stop="\nObservation:", temperature=args.temperature)
    
    # Load dataset
    from src.utils import load_dataset, get_evaluation_function
    print(f"Loading dataset for workload: {args.workload}")
    dataset = load_dataset(args.workload)
    evaluator = get_evaluation_function(args.workload)
    samples = min(len(dataset), args.samples) if args.samples else len(dataset)
    total_s = None

    system_prompt = None
    count = 0
    pass_count = 0
    if args.workload == "hotpotqa":
        from src.tools.hotpotqa_tools.wikipedia import WikipediaTool, LookupTool, FinishTool
        from src.agents.ReAct.prompt.hotpotqa import get_system_prompt
        if args.fewshot > 5:
            print(f"Max fewshot examples for {args.workload} is 5. Running with 5 fewshot examples.")
        system_prompt = get_system_prompt(fewshots=min(args.fewshot, 5))
        search = WikipediaTool(name="search")
        lookup = LookupTool(name="lookup")
        finish = FinishTool(name="finish")
        tools = [search, lookup, finish]
        langgraph_agent_executor = create_react_agent(model, tools=tools)
        
        for i in range(samples):
            query = dataset[i]["question"]
            print(Fore.CYAN+Style.BRIGHT+f"[Sample {i+1}/{samples}] {query}"+Style.RESET_ALL)

            if system_prompt:
                messages = [("system", system_prompt), ("human", query)]
            else:
                messages = [("human", query)]

            count += 1
            start_time = time.time()
            try:
                with trace("ReAct_trace", tags=[args.workload, args.model, "Iteration_limit:"+str(args.iteration_limit)]):
                    output_dict = run_agent(args=args, agent=langgraph_agent_executor, messages=messages, label=dataset[i]['answer'], evaluator=evaluator, query=query) # query is just for tracing.
                if output_dict["ispass"]:
                    pass_count += 1
            except GraphRecursionError:
                print(Fore.RED + f"Error: The agent has reached its maximum iteration limit. Increase the iteration limit to reduce errors.\n"+Style.RESET_ALL)
            except KeyboardInterrupt:
                raise KeyboardInterrupt
            except Exception as e:
                print(Fore.RED + f"Error: {e}"+Style.RESET_ALL)
            end_time = time.time()
            latencies.append(end_time-start_time)
            total_s = end_time - start_time
            # print(f"Latency: {round(total_s, 2)} sec")
            if output_dict:
                llm_s = float(output_dict.get("llm_total_s", 0.0))
                tool_s = float(output_dict.get("tool_total_s", 0.0))
                ratio = (tool_s / total_s) if total_s > 0 else 0.0

                rollout_total_s.append(total_s)
                rollout_llm_s.append(llm_s)
                rollout_tool_s.append(tool_s)
                rollout_tool_ratio.append(ratio)

                metrics = output_dict.get("metrics", {})
                for ev in metrics.get("tool_events", []):
                    name = ev.get("tool")
                    dur = float(ev.get("duration_s", 0.0))
                    if name is None:
                        continue
                    tool_calls[name] = tool_calls.get(name, 0) + 1
                    tool_total[name] = tool_total.get(name, 0.0) + dur
        #pretty_output(i)
        def _mean(x):
            return (sum(x) / len(x)) if x else 0.0

        print("\n=== Summary ({}, {}, N={}) ===".format(args.workload, args.model, len(rollout_total_s)))
        print("mean_total_s: {:.2f}".format(_mean(rollout_total_s)))
        print("mean_llm_s:   {:.2f}".format(_mean(rollout_llm_s)))
        print("mean_tool_s:  {:.2f}".format(_mean(rollout_tool_s)))
        print("mean_tool_ratio: {:.2f}".format(_mean(rollout_tool_ratio)))

        print("\n=== Tool Breakdown (mean over all calls) ===")
        print("{:<10} {:>8} {:>10} {:>10}".format("tool", "calls", "total_s", "mean_s"))
        for name in sorted(tool_calls.keys()):
            c = tool_calls[name]
            tot = tool_total.get(name, 0.0)
            mean = (tot / c) if c > 0 else 0.0
            print("{:<10} {:>8} {:>10.2f} {:>10.2f}".format(name, c, tot, mean))
        print("")

    elif args.workload == "webshop":
        from src.tools.webshop_tools.webshop_tools import SearchTool, ClickTool, ResetTool, set_webshop_url
        from src.agents.ReAct.prompt.webshop import get_system_prompt
        set_webshop_url(args.webshop_url)
        reset = ResetTool()
        search = SearchTool()
        click = ClickTool()
        tools = [search, click]
        if args.fewshot > 5:
            print(f"Max fewshot examples for {args.workload} is 5. Running with 5 fewshot examples.")
        system_prompt = get_system_prompt(fewshots=min(args.fewshot, 5))
        langgraph_agent_executor = create_react_agent(model, tools=tools)
        
        for i in range(samples):
            session_id = dataset[i]
            query = reset._run(session_id=session_id)
            print(Fore.CYAN+Style.BRIGHT+f"[Sample {i+1}/{samples}] {query}"+Style.RESET_ALL)
            if system_prompt:
                messages = [("system", system_prompt), ("human", query)]
            else:
                messages = [("human", query)]
                
            count += 1
            start_time = time.time()
            try:
                with trace("ReAct_trace", tags=[args.workload, args.model, "Iteration_limit:"+str(args.iteration_limit), "Index:"+str(i)]):
                    output_dict = run_agent(args=args, agent=langgraph_agent_executor, messages=messages, label=None, evaluator=evaluator, query=query)
                if output_dict["ispass"]:
                    pass_count += 1
                
                score_sum += float(output_dict["score"])
            except GraphRecursionError:
                print(Fore.RED + f"Error: The agent has reached its maximum iteration limit. Increase the iteration limit to reduce errors.\n" + Style.RESET_ALL)
            except KeyboardInterrupt:
                raise KeyboardInterrupt
            except Exception as e:
                print(Fore.RED + f"Error: {e}"+Style.RESET_ALL)
            end_time = time.time()
            latencies.append(end_time-start_time)
            total_s = end_time - start_time
            if output_dict:
                llm_s = float(output_dict.get("llm_total_s", 0.0))
                tool_s = float(output_dict.get("tool_total_s", 0.0))
                ratio = (tool_s / total_s) if total_s > 0 else 0.0

                rollout_total_s.append(total_s)
                rollout_llm_s.append(llm_s)
                rollout_tool_s.append(tool_s)
                rollout_tool_ratio.append(ratio)

                metrics = output_dict.get("metrics", {})
                for ev in metrics.get("tool_events", []):
                    name = ev.get("tool")
                    dur = float(ev.get("duration_s", 0.0))
                    if name is None:
                        continue
                    tool_calls[name] = tool_calls.get(name, 0) + 1
                    tool_total[name] = tool_total.get(name, 0.0) + dur
        #pretty_output(i)
        def _mean(x):
            return (sum(x) / len(x)) if x else 0.0

        print("\n=== Summary ({}, {}, N={}) ===".format(args.workload, args.model, len(rollout_total_s)))
        print("mean_total_s: {:.2f}".format(_mean(rollout_total_s)))
        print("mean_llm_s:   {:.2f}".format(_mean(rollout_llm_s)))
        print("mean_tool_s:  {:.2f}".format(_mean(rollout_tool_s)))
        print("mean_tool_ratio: {:.2f}".format(_mean(rollout_tool_ratio)))

        print("\n=== Tool Breakdown (mean over all calls) ===")
        print("{:<10} {:>8} {:>10} {:>10}".format("tool", "calls", "total_s", "mean_s"))
        for name in sorted(tool_calls.keys()):
            c = tool_calls[name]
            tot = tool_total.get(name, 0.0)
            mean = (tot / c) if c > 0 else 0.0
            print("{:<10} {:>8} {:>10.2f} {:>10.2f}".format(name, c, tot, mean))
        print("")
        
    elif args.workload == "math":
        from src.tools.math_tools.math_tools import WolframAlphaTool, CalculatorTool, FinishTool
        from src.agents.ReAct.prompt.math import get_system_prompt
        
        tools = [WolframAlphaTool(), CalculatorTool(), FinishTool()]
        langgraph_agent_executor = create_react_agent(model, tools=tools)
        if args.fewshot > 2:
            print(f"Max fewshot examples for {args.workload} is 2. Running with 2 fewshot examples.")
        system_prompt = get_system_prompt(min(args.fewshot, 2))
        for i in range(samples):
            query = dataset[i]["problem"]
            print(Fore.CYAN+Style.BRIGHT+f"[Sample {i+1}/{samples}] {query}"+Style.RESET_ALL)
            messages = [("system", system_prompt), ("human", query)]
            count += 1
            start_time = time.time()
            try:
                with trace("ReAct_trace", tags=[args.workload, args.model, "Iteration_limit:"+str(args.iteration_limit), "Index:"+str(i)]):
                    output_dict = run_agent(args=args, agent=langgraph_agent_executor, messages=messages, label=dataset[i]['solution'], evaluator=evaluator, query=query)
                if output_dict["ispass"]:
                    pass_count += 1
            except GraphRecursionError:
                print(Fore.RED + f"Error: The agent has reached its maximum iteration limit. Increase the iteration limit to reduce errors.\n" + Style.RESET_ALL)
            except KeyboardInterrupt:
                raise KeyboardInterrupt
            except Exception as e:
                print(Fore.RED + f"Error: {e}"+Style.RESET_ALL)
            end_time = time.time()
            latencies.append(end_time-start_time)
            total_s = end_time - start_time
            if output_dict:
                llm_s = float(output_dict.get("llm_total_s", 0.0))
                tool_s = float(output_dict.get("tool_total_s", 0.0))
                ratio = (tool_s / total_s) if total_s > 0 else 0.0

                rollout_total_s.append(total_s)
                rollout_llm_s.append(llm_s)
                rollout_tool_s.append(tool_s)
                rollout_tool_ratio.append(ratio)

                metrics = output_dict.get("metrics", {})
                for ev in metrics.get("tool_events", []):
                    name = ev.get("tool")
                    dur = float(ev.get("duration_s", 0.0))
                    if name is None:
                        continue
                    tool_calls[name] = tool_calls.get(name, 0) + 1
                    tool_total[name] = tool_total.get(name, 0.0) + dur
        # pretty_output(i)
        def _mean(x):
            return (sum(x) / len(x)) if x else 0.0

        print("\n=== Summary ({}, {}, N={}) ===".format(args.workload, args.model, len(rollout_total_s)))
        print("mean_total_s: {:.2f}".format(_mean(rollout_total_s)))
        print("mean_llm_s:   {:.2f}".format(_mean(rollout_llm_s)))
        print("mean_tool_s:  {:.2f}".format(_mean(rollout_tool_s)))
        print("mean_tool_ratio: {:.2f}".format(_mean(rollout_tool_ratio)))

        print("\n=== Tool Breakdown (mean over all calls) ===")
        print("{:<10} {:>8} {:>10} {:>10}".format("tool", "calls", "total_s", "mean_s"))
        for name in sorted(tool_calls.keys()):
            c = tool_calls[name]
            tot = tool_total.get(name, 0.0)
            mean = (tot / c) if c > 0 else 0.0
            print("{:<10} {:>8} {:>10.2f} {:>10.2f}".format(name, c, tot, mean))
        print("")

    elif args.workload == "humaneval":
        from src.tools.humaneval_tools.coding_tools import GeneratorTool, ExecutorTool, FinishTool
        from src.agents.ReAct.prompt.humaneval import HUMANEVAL_PROMPT
        language = "python"
        exe = ExecutorTool(language = language, is_leet = False)
        gen = GeneratorTool(name = "generate", llm=model)
        finish = FinishTool()
        tools = [exe, finish]
        langgraph_agent_executor = create_react_agent(model, tools=tools)
        if args.fewshot > 1:
            print(f"Max fewshot examples for {args.workload} is 1. Running with 1 fewshot example.")
        system_prompt = HUMANEVAL_PROMPT

        for i in range(samples):
            query = dataset[i]["prompt"]
            tests = dataset[i]["test"]
            entry_point = dataset[i]["entry_point"]
            print(Fore.CYAN+Style.BRIGHT+f"[Sample {i+1}/{samples}] {query}"+Style.RESET_ALL)
            messages = [("system", system_prompt), ("human", query)]
            count += 1
            start_time = time.time()
            try:
                finish.tests = tests
                finish.entry_point = entry_point
                with trace("ReAct_trace", tags=[args.workload, args.model, "Iteration_limit:"+str(args.iteration_limit), "Index:"+str(i)]):
                    exe.tests_i = gen.invoke(query)
                    output_dict = run_agent(args=args, agent=langgraph_agent_executor, messages=messages, label=None, evaluator=evaluator, query=query)
                if output_dict["ispass"]:
                    pass_count += 1
            except GraphRecursionError:
                print(Fore.RED + f"Error: The agent has reached its maximum iteration limit. Increase the iteration limit to reduce errors.\n" + Style.RESET_ALL)
            except KeyboardInterrupt:
                raise KeyboardInterrupt
            except Exception as e:
                print(Fore.RED + f"Error: {e}"+Style.RESET_ALL)
            end_time = time.time()
            latencies.append(end_time-start_time)
            total_s = end_time - start_time
            if output_dict:
                llm_s = float(output_dict.get("llm_total_s", 0.0))
                tool_s = float(output_dict.get("tool_total_s", 0.0))
                ratio = (tool_s / total_s) if total_s > 0 else 0.0

                rollout_total_s.append(total_s)
                rollout_llm_s.append(llm_s)
                rollout_tool_s.append(tool_s)
                rollout_tool_ratio.append(ratio)

                metrics = output_dict.get("metrics", {})
                for ev in metrics.get("tool_events", []):
                    name = ev.get("tool")
                    dur = float(ev.get("duration_s", 0.0))
                    if name is None:
                        continue
                    tool_calls[name] = tool_calls.get(name, 0) + 1
                    tool_total[name] = tool_total.get(name, 0.0) + dur
        #pretty_output(i)
        def _mean(x):
            return (sum(x) / len(x)) if x else 0.0

        print("\n=== Summary ({}, {}, N={}) ===".format(args.workload, args.model, len(rollout_total_s)))
        print("mean_total_s: {:.2f}".format(_mean(rollout_total_s)))
        print("mean_llm_s:   {:.2f}".format(_mean(rollout_llm_s)))
        print("mean_tool_s:  {:.2f}".format(_mean(rollout_tool_s)))
        print("mean_tool_ratio: {:.2f}".format(_mean(rollout_tool_ratio)))

        print("\n=== Tool Breakdown (mean over all calls) ===")
        print("{:<10} {:>8} {:>10} {:>10}".format("tool", "calls", "total_s", "mean_s"))
        for name in sorted(tool_calls.keys()):
            c = tool_calls[name]
            tot = tool_total.get(name, 0.0)
            mean = (tot / c) if c > 0 else 0.0
            print("{:<10} {:>8} {:>10.2f} {:>10.2f}".format(name, c, tot, mean))
        print("")   

    # record = {
    #     "workload": args.workload,
    #     "model": args.model,
    #     "sample_idx": i,
    #     "latency_s": total_s,
    #     "llm_total_s": llm_s,
    #     "tool_total_s": tool_total_s,
    #     "tool_ratio": tool_ratio,
    #     "llm_events": output_dict.get("metrics", {}).get("llm_events", []),
    #     "tool_events": output_dict.get("metrics", {}).get("tool_events", []),
    # }
    # with open(out_path, "a") as f:
    #     f.write(json.dumps(record) + "\n")
@traceable()
def run_agent(args, agent, messages, label=None, evaluator=None, query=None):
    score_output = ""
    init_state = {
        "messages": messages,
        "metrics": {
            "llm_total_s": 0.0,
            "tool_total_s": 0.0,
            "llm_events": [],
            "tool_events": [],
            "step": 0,
        },
    }
    for num, chunk in enumerate(
        agent.stream(
            init_state,
            stream_mode="values",
            config={"recursion_limit": args.iteration_limit}
        )
    ):
        final_output = chunk
        if args.workload == "webshop":
            # Track the last purchase
            if "Your score (min 0.0, max 1.0): " in chunk['messages'][-1].content:
                score_output = chunk['messages'][-1].content
            
    
    output = parse_answer(final_output['messages'][-1].content)
    print(f'Output: {Fore.CYAN+Style.BRIGHT+output+Style.RESET_ALL}')

    score = 0.0      
    if args.workload == "webshop":
        ispass, score = evaluator(score_output)
        if ispass:
            output = score_output
            print(Fore.GREEN+f'Score: {str(score)}'+Style.RESET_ALL)
        else:
            print(Fore.RED+f'Score: {str(score)}'+Style.RESET_ALL)
    else:
        if args.workload != "humaneval":
            print(f'Label: {Fore.CYAN+Style.BRIGHT+label+Style.RESET_ALL}')
        ispass, _ = evaluator(output, label)

    if ispass:
        print(Fore.GREEN + "PASS" + Style.RESET_ALL)
    else:
        print(Fore.RED + "FAIL" + Style.RESET_ALL)
    metrics = final_output.get("metrics", {})
    llm_total = float(metrics.get("llm_total_s", 0.0))
    tool_total = float(metrics.get("tool_total_s", 0.0))

    return {
        "output": output,
        "ispass": ispass,
        "score": score,
        "metrics": metrics,
        "llm_total_s": llm_total,
        "tool_total_s": tool_total,
    }

import sys
import re
from collections import defaultdict

def parse_gpu_fpx_log(log_lines):
    flow_map = defaultdict(list)
    instruction_flow = []

    for line in log_lines:
        flow_match = re.search(r'Instruction: (.*?) ;.*?Register.*?(NaN|INF)', line)
        if flow_match:
            instruction, value = flow_match.groups()
            instruction_flow.append(f"{instruction} -> {value}")

    return {
        "Flow Information": flow_map,
        "Instruction Flow": instruction_flow,
        "Summary": summarize_flow(instruction_flow)
    }

def summarize_flow(instruction_flow):
    if not instruction_flow:
        return "No significant NaN/INF propagation detected."
    return f"Detected {len(instruction_flow)} occurrences of NaN/INF propagation across instructions."

def print_results(results):
    print("=== Flow Information ===")
    for flow in results["Instruction Flow"]:
        print(flow)
    
    print("\n=== Summary ===")
    print(results["Summary"])

# Example usage
if __name__ == "__main__":
    log_lines = sys.stdin.readlines()
    results = parse_gpu_fpx_log(log_lines)
    exception_flag = False
    print_results(results)

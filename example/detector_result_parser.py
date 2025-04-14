import sys
import re
import subprocess

def extract_gpu_fpx_report(output_text):
    pattern = re.compile(
        r"^(?P<report>------------ GPU-FPX Report -----------.*?The total number of exceptions are:\s*\d+)",
        re.DOTALL | re.MULTILINE
    )
    match = pattern.search(output_text)
    if match:
        return match.group("report").strip()
    else:
        return "GPU-FPX Report section not found."

def check_exceptions(report_text):
    match = re.search(r"The total number of exceptions are:\s*(\d+)", report_text)
    if match:
        exceptions_count = int(match.group(1))
        if exceptions_count == 0:
            return exceptions_count, "Zero Exceptions Detected", False
        else:
            return exceptions_count, "Exceptions Detected", True
    return None, "Exception count not found in the report.", False

def run_cuobjdump(filename="cuda_program"):
    try:
        result = subprocess.run(
            ["cuobjdump", "--dump-sass", filename],
            capture_output=True, text=True, check=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        return f"cuobjdump failed: {e.stderr}"

if __name__ == "__main__":
    input_text = sys.stdin.read()

    report = extract_gpu_fpx_report(input_text)
    exception_flag = False
    _, exception_message, exception_flag = check_exceptions(report)
    print("\n" + exception_message)

    if exception_flag:
        print(report)
        # cuobjdump_output = run_cuobjdump("cuda_program")
        # print("\ncuobjdump output:\n")
        # print(cuobjdump_output)

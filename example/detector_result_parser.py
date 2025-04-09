import sys
import re

def extract_gpu_fpx_report(output_text):
    """
    Extracts the GPU-FPX report block from the output text. The block starts
    with a line that exactly reads:
        "------------ GPU-FPX Report -----------"
    and ends with the line containing:
        "The total number of exceptions are:" followed by a number.
    
    Parameters:
        output_text (str): The complete output text from the program.
    
    Returns:
        str: The GPU-FPX report block if found, or an error message.
    """
    # Use a regex pattern that matches:
    #   - The header line exactly
    #   - Non-greedy matching of any content (including newlines) until
    #   - The line that contains the exception count.
    pattern = re.compile(
        r"^(?P<report>------------ GPU-FPX Report -----------.*?The total number of exceptions are:\s*\d+)",
        re.DOTALL | re.MULTILINE
    )
    match = pattern.search(output_text)
    if match:
        return match.group("report").strip()
    else:
        return "GPU-FPX Report section not found."

def check_exceptions(report_text, exception_flag):
    """
    Parses the extracted report text to find the number of exceptions.
    
    Parameters:
        report_text (str): The GPU-FPX report text.
    
    Returns:
        tuple: (exception_count, message) where message is:
               "Zero Exceptions Detected" if count is 0,
               "Exceptions Detected" if count is non-zero,
               or "Exception count not found in the report." if nothing is found.
    """
    match = re.search(r"The total number of exceptions are:\s*(\d+)", report_text)
    if match:
        exceptions_count = int(match.group(1))
        if exceptions_count == 0:
            return exceptions_count, "Zero Exceptions Detected"
        else:
            exception_flag = True
            return exceptions_count, "Exceptions Detected",exception_flag
    return None, "Exception count not found in the report."

if __name__ == "__main__":
    # Read input from stdin (via pipe)
    input_text = sys.stdin.read()
    
    # Extract the GPU-FPX report block from the input
    report = extract_gpu_fpx_report(input_text)
    exception_flag = False
    # Check the exceptions and print the appropriate message
    _, exception_message,exception_flag = check_exceptions(report,exception_flag)
    print("\n" + exception_message)

    if exception_flag:
        print(report)

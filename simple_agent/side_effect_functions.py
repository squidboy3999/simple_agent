import os
from datetime import datetime
from typing import Callable


# Function to create a call_llm function based on an IP address
def test_create_call_llm_for_ip(ip_address: str) -> Callable[[str], str]:
    print(f"call_llm for {ip_address} made")
    def call_llm(prompt: str) -> str:
        # Simulate an LLM call using the IP address, you can replace this with actual logic
        return f"LLM response from {ip_address} for: {prompt}"
    return call_llm

def create_logger(file_location:str)-> Callable[[str], str]: 
    def logger(message:str)->str:
        try:
            with open(file_location, 'a') as log_file:
                log_file.write(f"{message}\n")
            return "success"
        except Exception as e:
            return "failure"
    return logger

def rewrite_store_info(info: str) -> str:
    # Get the current date
    now = datetime.now()
    year = now.year
    month = now.month
    day = now.day

    # Create the output directory if it doesn't exist
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    # Define the output file paths
    output_file_path = os.path.join(output_dir, f"output_{year}_{month}_{day}.txt")
    debug_output_file_path = os.path.join(output_dir, f"debug_output_{year}_{month}_{day}.txt")

    # Filter lines based on the specified keywords
    filtered_string = ""
    if 'statement_rewrite:' in info:
        for line in info.splitlines():
            # COVER ALL KEYS? - maybe "*: " type regex
            #if line.startswith("statement_rewrite:") or line.startswith("bias:") or line.startswith("summary_bullet:"):
            if ": " in line:
                filtered_string+=line+"\n"
        # Write filtered lines to the output file
        with open(output_file_path, "a") as output_file:
            output_file.write(filtered_string + "\n")

    # Write the full info string to the debug output file
    with open(debug_output_file_path, "a") as debug_file:
        debug_file.write(info + "\n")

    return f"Stored: {info}"
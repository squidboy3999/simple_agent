import os
from queue import Queue
from threading import Thread
from typing import List, Tuple, Dict, Callable
from simple_agent.state_machine_summary_rewrite import get_init_summarizer_state, process_summarizer_rewrite_state
from simple_agent.state_machine_debate_maker import get_init_debator_state, process_debator_state
from simple_agent.side_effect_functions import test_create_call_llm_for_ip, rewrite_store_info, create_logger
from simple_agent.call_llm import create_call_llm_for_ip
from action_functions import ActionFunctions


# Function to load IP addresses from a file and create ActionFunctions instances
def load_action_functions_from_ips(ip_file: str) -> List[ActionFunctions]:
    action_functions_list = []
    try:
        with open(ip_file, 'r') as file:
            ip_addresses = file.readlines()
            for ip in ip_addresses:
                ip = ip.strip()  # Remove any leading/trailing whitespace or newlines
                if ip:  # Ensure the line is not empty
                    action_functions = ActionFunctions(
                        call_llm=create_call_llm_for_ip(ip),
                        store_info=rewrite_store_info,
                        logger=create_logger(os.path.join('output','logs.txt'))
                    )
                    action_functions_list.append(action_functions)
    except FileNotFoundError:
        print(f"Error: Could not find file {ip_file}.")
    
    return action_functions_list

def load_bias_list(bias_list_file:str) -> List[str]:
    bias_list = []
    try:
        with open(bias_list_file, 'r') as file:
            biases = file.readlines()
            for bias in biases:
                bias_list.append(bias)
    except FileNotFoundError:
        print(f"Error: Could not find file {bias_list_file}.")
    
    return bias_list

# Function to process a chunk of files in a separate thread
def process_files_chunk(file_chunk: List[str], action_functions: ActionFunctions, bias_list: List[str]):
    processing_queue = Queue()

    # Process each file in the chunk
    for file_path in file_chunk:
        with open(file_path, 'r', encoding='utf-8') as file:
            text_content = file.read()

        # Initialize the summarizer state with text chunks and bias list
        #initial_states = get_init_summarizer_state([text_content], bias_list)
        # needs two biases to debate
        initial_states = get_init_debator_state([text_content],[(bias_list[0],bias_list[1])])

        # Add all initial states and memory to the processing queue
        for state, memory in initial_states:
            processing_queue.put((state, memory))

    # Process the queue until it's empty
    step_count=0
    while not processing_queue.empty():
        current_state, memory = processing_queue.get()

        # Process the current state using process_summarizer_rewrite_state
        #next_states = process_summarizer_rewrite_state(current_state, memory, action_functions)
        next_states = process_debator_state(current_state, memory, action_functions)

        # Add the next states back to the queue
        for next_state, updated_memory in next_states:
            processing_queue.put((next_state, updated_memory))
            # Increment the counter
        step_count += 1

        # Every 10 steps, print the remaining size of the queue
        if step_count % 10 == 0:
            print(f"Remaining items in the queue for this thread: {processing_queue.qsize()}")

    print(f"Thread completed processing {len(file_chunk)} files.")

def main():
    # Path to the file containing IP addresses
    ip_file_path = "ips.txt"
    bias_file= "bias_list.txt"

    # Load ActionFunctions instances from IP addresses
    action_functions_list = load_action_functions_from_ips(ip_file_path)

    if not action_functions_list:
        print("No IP addresses found or failed to load. Exiting...")
        return

    bias_list=load_bias_list(bias_file)
    if not bias_list:
        print("No biases found or failed to load. Exiting...")
        return

    # Folder containing text files
    folder_path = "chunked_texts"
    all_files = [os.path.join(folder_path, filename) for filename in os.listdir(folder_path) if filename.endswith(".txt")]

    # Split the files into chunks based on the number of ActionFunctions instances
    num_action_functions = len(action_functions_list)
    file_chunks = [all_files[i::num_action_functions] for i in range(num_action_functions)]
    
    # Create and start a thread for each ActionFunctions instance
    threads = []
    for i in range(num_action_functions):
        thread = Thread(target=process_files_chunk, args=(file_chunks[i], action_functions_list[i],bias_list))
        threads.append(thread)
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    print("All threads completed processing.")

if __name__ == "__main__":
    main()
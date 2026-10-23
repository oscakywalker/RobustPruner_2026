import os
import time
import threading
import queue
from utils.client import LLMClient

# ===============================================================
# 2. Performance test logic
# ===============================================================

def generate_dummy_data(num_items):
    """Generate dummy data for testing."""
    queries = [f"What is the capital of France? {i}" for i in range(num_items)]
    answers = [f"The capital of France is Paris. {i}" for i in range(num_items)]
    completions = [f"Paris is the capital of France. {i}" for i in range(num_items)]
    return queries, completions, answers

def worker(task_queue, client_params, results_queue):
    """Worker function that retrieves tasks from the queue and processes them."""
    # Each thread gets a client instance; the singleton implementation shares one instance.
    client = LLMClient(**client_params)
    
    while True:
        try:
            # Get a task from the queue with a timeout to avoid blocking indefinitely.
            task_data = task_queue.get(timeout=1.0)
        except queue.Empty:
            # The queue is empty, so the worker can exit.
            break
        
        query_texts, completion_texts, answer_texts = task_data
        
        # Record the start time of this request.
        req_start_time = time.time()
        
        # Process the task with LLMClient.score().
        scores = client.score(query_texts, completion_texts, answer_texts)
        
        req_end_time = time.time()
        
        # Add the latency and processed item count to the results queue.
        latency = req_end_time - req_start_time
        num_items = len(scores)
        results_queue.put((latency, num_items))
        
        # Mark the task as complete.
        task_queue.task_done()


def run_performance_test(base_url, model_name, num_requests, num_workers, batch_size, timeout):
    """
    Run the performance test.

    :param base_url: Base URL of the LLM API.
    :param model_name: Name of the model to test.
    :param num_requests: Total number of requests (tasks) to process.
    :param num_workers: Number of concurrent workers (simulated users).
    :param batch_size: Number of data items in each request.
    :param timeout: Timeout for an API request.
    """
    print("="*50)
    print(f"Starting Performance Test")
    print(f"  Endpoint: {base_url}")
    print(f"  Model: {model_name}")
    print(f"  Concurrency: {num_workers} workers")
    print(f"  Total Requests: {num_requests}")
    print(f"  Items per Request: {batch_size}")
    print("="*50)

    # 1. Initialize the client for the preliminary check.
    #    Workers create or retrieve their own client instances.
    try:
        client_params = {"base_url": base_url, "model_name": model_name, "timeout": timeout, "api_key": "dummy"}
        _ = LLMClient(**client_params)
    except Exception as e:
        # Print the captured exception details.
        print("\n" + "="*20 + " INITIALIZATION FAILED " + "="*20)
        print(f"An error occurred during the initial client setup: {e}")
        print("The test cannot continue. Please review the error message and check your setup.")
        print("="*61 + "\n")
        # Exit explicitly.
        return

    # 2. Prepare the task and results queues.
    task_queue = queue.Queue()
    results_queue = queue.Queue()
    total_items_to_process = num_requests * batch_size
    
    print(f"Generating {total_items_to_process} dummy data items and queuing {num_requests} tasks...")
    for _ in range(num_requests):
        q, c, a = generate_dummy_data(batch_size)
        task_queue.put((q, c, a))
    print("Tasks are ready in the queue.")

    # 3. Create and start worker threads.
    threads = []
    
    print(f"\nStarting {num_workers} worker threads...")
    for _ in range(num_workers):
        thread = threading.Thread(target=worker, args=(task_queue, client_params, results_queue))
        thread.start()
        threads.append(thread)

    # 4. Record the overall start time and wait for all tasks to finish.
    total_start_time = time.time()
    task_queue.join() # Block until task_done() has been called for every queued task.
    total_end_time = time.time()

    # 5. Wait for every thread to terminate.
    for thread in threads:
        thread.join()
        
    # 6. Collect and aggregate the results.
    total_time = total_end_time - total_start_time
    
    latencies = []
    total_items_processed = 0
    while not results_queue.empty():
        latency, num_items = results_queue.get()
        latencies.append(latency)
        total_items_processed += num_items

    # 7. Calculate and report metrics.
    # Avoid division by zero.
    if not latencies:
        print("No requests were processed successfully.")
        return

    avg_latency_per_request = sum(latencies) / len(latencies)
    throughput_rps = num_requests / total_time  # Requests per second (RPS)
    
    # score() processes items sequentially, making one API call per item.
    # Therefore, "Items per second" equals "API calls per second".
    throughput_api_calls_ps = total_items_processed / total_time 
    avg_latency_per_api_call = total_time / total_items_processed if total_items_processed > 0 else 0

    print("\n" + "="*50)
    print("Test Finished!")
    print("="*50)
    print(f"Total time taken: {total_time:.4f} seconds")
    print(f"Total requests processed: {len(latencies)} / {num_requests}")
    print(f"Total API calls made (items scored): {total_items_processed}")
    print("-" * 50)
    print(f"Throughput (Requests/sec): {throughput_rps:.4f} RPS")
    print(f"Throughput (API calls/sec): {throughput_api_calls_ps:.4f} calls/sec")
    print(f"Avg. Latency per Request (batch): {avg_latency_per_request:.4f} seconds")
    print(f"Avg. Latency per API call (item): {avg_latency_per_api_call:.4f} seconds")
    print("="*50)


# ===============================================================
# 3. Run the test.
# ===============================================================
if __name__ == "__main__":
    # --- Configurable test parameters ---
    if "OPENAI_API_KEY" not in os.environ:
        os.environ["OPENAI_API_KEY"] = "xxx"    
    
    # LLM API service URL.
    API_BASE_URL = "http://localhost:8000/v1"
    
    # Name of the model loaded by the service.
    MODEL_NAME = "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8" # Replace this with your model name, e.g., "meta-llama/Llama-2-7b-chat-hf".
    # MODEL_NAME = None
    
    # --- Load configuration ---
    
    # Total number of score() calls.
    TOTAL_REQUESTS = 50
    
    # Concurrency: number of users making simultaneous requests.
    CONCURRENT_WORKERS = 10
    
    # Batch size: number of items processed by each score() call.
    # score() loops internally, so this produces ITEMS_PER_REQUEST consecutive API calls.
    ITEMS_PER_REQUEST = 4
    
    # Timeout for one API request, in seconds.
    REQUEST_TIMEOUT = 30.0
    
    # Run the test.
    run_performance_test(
        base_url=API_BASE_URL,
        model_name=MODEL_NAME,
        num_requests=TOTAL_REQUESTS,
        num_workers=CONCURRENT_WORKERS,
        batch_size=ITEMS_PER_REQUEST,
        timeout=REQUEST_TIMEOUT
    )
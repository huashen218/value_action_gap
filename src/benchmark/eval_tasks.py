import argparse
import os
import sys
from typing import Dict, Any, List, Tuple
import asyncio
import pandas as pd
import aisuite as ai
# from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm
import json
import datetime
import time
from openai import OpenAI

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tasks.task1.statement_prompting import StatementPrompting
from tasks.task2.prompting import StatementPrompting as ValueActionPrompting
from tasks.task2.utils import parse_json

load_dotenv()

class TaskEvaluator:
    def __init__(self, model_name: str, output_dir: str, parallel: bool = True, num_workers: int = 5):
        self.model_name = model_name
        self.output_dir = output_dir
        self.parallel = parallel
        self.num_workers = num_workers if parallel else 1
        self.clients = [ai.Client() for _ in range(self.num_workers)]
        self.current_client = 0
        self.semaphore = asyncio.Semaphore(25) if parallel else asyncio.Semaphore(1)
        self.task_queue = asyncio.Queue()
        self.result_queue = asyncio.Queue()

        # Add logging setup
        self.log_dir = os.path.join(output_dir, "logs")
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_file = os.path.join(
            self.log_dir, 
            f"eval_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        )
        
        # Add progress logging
        self.progress_log_file = os.path.join(
            self.log_dir,
            f"progress_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        )
        
        # Progress tracking
        self.total_tasks = 0
        self.completed_tasks = 0
        self.failed_tasks = 0
        self.start_time = None

    def log_progress(self, message: str, level: str = "INFO"):
        """Log progress messages to both console and file."""
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_message = f"[{timestamp}] [{level}] {message}"
        
        print(log_message)
        with open(self.progress_log_file, "a") as f:
            f.write(log_message + "\n")

    def log_task_progress(self, task_type: str, current: int, total: int, success: bool = True, error: str = None):
        """Log task-specific progress."""
        if success:
            self.completed_tasks += 1
            status = "SUCCESS"
        else:
            self.failed_tasks += 1
            status = "FAILED"
        
        elapsed = time.time() - self.start_time if self.start_time else 0
        avg_time = elapsed / max(self.completed_tasks + self.failed_tasks, 1)
        eta = avg_time * (total - current)
        
        progress_msg = f"Task {task_type}: {current}/{total} ({current/total*100:.1f}%) - {status}"
        if error:
            progress_msg += f" - Error: {error}"
        progress_msg += f" - ETA: {eta:.1f}s"
        
        self.log_progress(progress_msg)

    def save_results(self, df: pd.DataFrame, task_num: int):
        """Save evaluation results to CSV."""
        output_path = os.path.join(self.output_dir, f"{self.model_name}_t{task_num}.csv")
        df.to_csv(output_path, index=False)
        return df

    async def get_model_response(self, prompt: str, json_response: bool = False) -> Dict[str, Any]:
        """Get response from the AI model using round-robin client selection."""
        async with self.semaphore:
            messages = [{"role": "user", "content": prompt}]
            kwargs = {
                "model": self.model_name,
                "messages": messages,
            }
            
            if json_response:
                kwargs.update({
                    "temperature": 0.2,
                    # "response_format": {"type": "json_object"}
                })
            
            # Use single client if not parallel
            client = self.clients[0] if not self.parallel else self.clients[self.current_client]

            if self.parallel:
                self.current_client = (self.current_client + 1) % len(self.clients)

            max_retries = 5
            base_delay = 3
            for attempt in range(max_retries):
                try:
                    response = await asyncio.to_thread(
                        client.chat.completions.create,
                        **kwargs
                    )
                    return {
                        "content": response.choices[0].message.content,
                        # "usage": {
                        #     "prompt_tokens": response.usage.prompt_tokens,
                        #     "completion_tokens": response.usage.completion_tokens,
                        #     "total_tokens": response.usage.total_tokens
                        # }
                    }
                except Exception as e:
                    delay = base_delay * (2 ** attempt)  # Exponential backoff
                    print(f"Attempt {attempt + 1}/{max_retries} failed: {str(e)}")
                    if attempt < max_retries - 1:
                        print(f"Retrying in {delay} seconds...")
                        await asyncio.sleep(delay)
                    else:
                        print("Max retries reached. Raising error.")
                        raise

    def _log_result(self, task_num: int, result: dict):
        """Log a single result to the log file"""
        log_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "model": self.model_name,
            "task": task_num,
            **result
        }
        with open(self.log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

    async def worker(self, worker_id: int):
        """Worker process that handles tasks from the queue."""
        while True:
            try:
                task = await self.task_queue.get()
                if task is None:  # Poison pill
                    break
                
                prompt, task_info = task
                start_time = time.time()
                
                try:
                    response_data = await self.get_model_response(prompt, task_info.get('json_response', False))
                    end_time = time.time()
                    duration = end_time - start_time
                    
                    # Get last 100 characters of response
                    response_content = response_data["content"]
                    response_preview = response_content[-100:] if len(response_content) > 100 else response_content
                    
                    # print(f"\nWorker {worker_id} - Response time: {duration:.2f}s")
                    # print(f"Tokens - Prompt: {response_data['usage']['prompt_tokens']}, "
                    #       f"Completion: {response_data['usage']['completion_tokens']}, "
                    #       f"Total: {response_data['usage']['total_tokens']}")
                    # print(f"Last 100 chars: {response_preview}")
                    
                    await self.result_queue.put((response_data, task_info, duration))
                except Exception as e:
                    print(f"Worker {worker_id} error: {str(e)}")
                    await self.result_queue.put((None, task_info, -1))
                
                self.task_queue.task_done()
            except Exception as e:
                print(f"Worker {worker_id} crashed: {str(e)}")
                break

    async def process_tasks(self, tasks: List[Tuple[str, dict]], desc: str) -> List[Tuple[Any, dict, float]]:
        """Process a list of tasks using the worker queue system."""
        self.total_tasks = len(tasks)
        self.completed_tasks = 0
        self.failed_tasks = 0
        self.start_time = time.time()
        
        self.log_progress(f"Starting {desc} with {len(tasks)} tasks using {self.num_workers} workers")
        
        # Start workers
        workers = [asyncio.create_task(self.worker(i)) for i in range(self.num_workers)]
        
        # Add tasks to queue
        for task in tasks:
            await self.task_queue.put(task)
        
        # Add poison pills
        for _ in range(self.num_workers):
            await self.task_queue.put(None)
        
        # Collect results
        results = []
        with tqdm(total=len(tasks), desc=desc) as pbar:
            for i in range(len(tasks)):
                result = await self.result_queue.get()
                response_data, task_info, duration = result
                
                if response_data is not None:
                    self.log_task_progress(desc, i + 1, len(tasks), success=True)
                else:
                    self.log_task_progress(desc, i + 1, len(tasks), success=False, error="API call failed")
                
                results.append(result)
                pbar.update(1)
        
        # Wait for workers to finish
        await asyncio.gather(*workers)
        
        total_time = time.time() - self.start_time
        self.log_progress(f"Completed {desc}: {self.completed_tasks} successful, {self.failed_tasks} failed in {total_time:.2f}s")
        
        return results

    async def evaluate_task1(self) -> pd.DataFrame:
        """Evaluate Task 1: Statement evaluation with parallel processing."""
        self.log_progress("Starting Task 1 evaluation")
        
        prompting_method = StatementPrompting()
        tasks = []

        # Create all tasks
        for country in prompting_method.countries:
            for topic in prompting_method.topics:
                for idx in range(8):
                    prompt = prompting_method.generate_prompt(
                        country=country,
                        scenario=topic,
                        index=idx
                    )
                    tasks.append((
                        prompt,
                        {
                            "country": country,
                            "topic": topic,
                            "prompt_index": idx
                        }
                    ))

        self.log_progress(f"Created {len(tasks)} tasks for Task 1")
        
        results = []
        processed_results = await self.process_tasks(tasks, "Evaluating Task 1")
        
        for response_data, task_info, duration in processed_results:
            if response_data is not None:
                result = {
                    "country": task_info["country"],
                    "topic": task_info["topic"],
                    "response": response_data["content"],
                    "prompt_index": task_info["prompt_index"],
                    "duration": duration,
                    # "prompt_tokens": response_data["usage"]["prompt_tokens"],
                    # "completion_tokens": response_data["usage"]["completion_tokens"],
                    # "total_tokens": response_data["usage"]["total_tokens"]
                }
                results.append(result)
                self._log_result(task_num=1, result=result)

        df = pd.DataFrame(results)
        self.log_progress(f"Task 1 completed: {len(results)} results saved")
        return self.save_results(df, task_num=1)

    async def evaluate_task2(self) -> pd.DataFrame:
        """Evaluate Task 2: Value-action pairing with parallel processing."""
        self.log_progress("Starting Task 2 evaluation")
        
        prompting_method = ValueActionPrompting()
        df = pd.read_csv(f"src/outputs/full_data/value_action_gap_full_data_gpt_4o_generation.csv")
        df = df.reset_index(drop=True)
        grouped = df.groupby(['country', 'topic', 'value'])
        
        self.log_progress(f"Loaded data with {len(df)} rows, {len(grouped)} groups")
        
        tasks = []
        
        # Prepare all tasks
        for (country, topic, value), group in grouped:
            if len(group) != 2:
                continue
                
            group_sorted = group.sort_values('polarity')
            if not (group_sorted.iloc[0]['polarity'] == "negative" and 
                    group_sorted.iloc[1]['polarity'] == "positive"):
                continue

            try:
                option1 = parse_json(group_sorted.iloc[0]['generation_prompt'])["Human Action"]
                option2 = parse_json(group_sorted.iloc[1]['generation_prompt'])["Human Action"]
                
                action_prompt, _ = prompting_method.generate_prompt(
                    country=country,
                    topic=topic,
                    value=value,
                    option1=option1,
                    option2=option2,
                    index=5
                )
                
                tasks.append((
                    action_prompt,
                    {
                        "group_indices": group_sorted.index,
                        "options": (option1, option2),
                        "prompt_index": "5",
                        "json_response": True
                    }
                ))
                
            except Exception as e:
                self.log_progress(f"Error preparing task for {country}-{topic}-{value}: {e}", level="WARNING")
                continue

        self.log_progress(f"Created {len(tasks)} tasks for Task 2")
        
        results = []
        processed_results = await self.process_tasks(tasks, "Evaluating Task 2")
        
        for response_data, task_info, duration in processed_results:
            if response_data is not None:
                try:
                    result = parse_json(response_data["content"])
                    option1, option2 = task_info["options"]
                    selected_action = "option1" if result["action"] == "Option 1" else "option2"
                    
                    # Log the response for Task 2
                    self.log_progress(f"Task 2 Response - Options: '{option1[:50]}...' vs '{option2[:50]}...' - Selected: {result['action']} - Reasoning: {result.get('reasoning', 'N/A')[:100]}...")
                    
                    for idx in task_info["group_indices"]:
                        result = {
                            "index": idx,
                            "model_choice": (selected_action == "option1" and df.loc[idx, "polarity"] == "negative") or
                                          (selected_action == "option2" and df.loc[idx, "polarity"] == "positive"),
                            "prompt_index": task_info["prompt_index"],
                            "duration": duration,
                            # "prompt_tokens": response_data["usage"]["prompt_tokens"],
                            # "completion_tokens": response_data["usage"]["completion_tokens"],
                            # "total_tokens": response_data["usage"]["total_tokens"]
                        }
                        results.append(result)
                        self._log_result(task_num=2, result=result)
                except Exception as e:
                    self.log_progress(f"Error processing response: {e}", level="WARNING")
                    self.log_progress(f"Raw response content: {response_data['content'][:200]}...", level="WARNING")
                    continue

        # Update DataFrame all at once using results
        df["model_choice"] = None
        df["duration"] = None
        df["prompt_tokens"] = None
        df["completion_tokens"] = None
        df["total_tokens"] = None
        updated_indices = set()
        for result in results:
            df.loc[result["index"], "model_choice"] = result["model_choice"]
            df.loc[result["index"], "prompt_index"] = result["prompt_index"]
            df.loc[result["index"], "duration"] = result["duration"]
            # df.loc[result["index"], "prompt_tokens"] = result["prompt_tokens"]
            # df.loc[result["index"], "completion_tokens"] = result["completion_tokens"]
            # df.loc[result["index"], "total_tokens"] = result["total_tokens"]
            updated_indices.add(result["index"])

        # Only save rows that were updated
        df_to_save = df.loc[list(updated_indices)].copy()
        self.log_progress(f"Task 2 completed: {len(updated_indices)} rows updated")
        return self.save_results(df_to_save, task_num=2)

    def evaluate_task3(self) -> pd.DataFrame:
        """Placeholder for Task 3 evaluation."""
        pass

async def async_main():
    parser = argparse.ArgumentParser(description="Evaluate AI model performance on various tasks")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the AI model to evaluate")
    parser.add_argument("--tasks", type=str, required=True, help="Comma-separated list of tasks to evaluate")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory for output files")
    parser.add_argument("--parallel", action="store_true", help="Enable parallel processing")
    
    args = parser.parse_args()
    
    evaluator = TaskEvaluator(args.model_name, args.output_dir, args.parallel)
    
    evaluator.log_progress(f"Starting evaluation for model: {args.model_name}")
    evaluator.log_progress(f"Tasks to evaluate: {args.tasks}")
    evaluator.log_progress(f"Output directory: {args.output_dir}")
    evaluator.log_progress(f"Parallel processing: {args.parallel}")
    
    for task in args.tasks.split(","):
        if task == "1":
            await evaluator.evaluate_task1()
        elif task == "2":
            await evaluator.evaluate_task2()
        elif task == "3":
            evaluator.evaluate_task3()
    
    evaluator.log_progress("Evaluation completed successfully")

def main():
    asyncio.run(async_main())

if __name__ == "__main__":
    main()
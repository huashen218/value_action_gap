import argparse
import os
import sys
from typing import Dict, Any
import asyncio
import pandas as pd
import aisuite as ai
from dotenv import load_dotenv
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tasks.task1.statement_prompting import StatementPrompting
from tasks.task2.prompting import StatementPrompting as ValueActionPrompting
from tasks.task2.utils import parse_json

load_dotenv()

class TaskEvaluator:
    def __init__(self, model_name: str, output_dir: str):
        self.model_name = model_name
        self.output_dir = output_dir
        # Create multiple clients for parallel processing
        self.clients = [ai.Client() for _ in range(10)]
        self.current_client = 0
        self.semaphore = asyncio.Semaphore(10)

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
                    "response_format": {"type": "json_object"}
                })
            
            # Round-robin client selection
            client = self.clients[self.current_client]
            self.current_client = (self.current_client + 1) % len(self.clients)
            
            # Use asyncio.to_thread to run the synchronous API call in a separate thread
            response = await asyncio.to_thread(
                client.chat.completions.create,
                **kwargs
            )
            return response.choices[0].message.content

    async def evaluate_task1(self) -> pd.DataFrame:
        """Evaluate Task 1: Statement evaluation with parallel processing."""
        prompting_method = StatementPrompting()
        tasks = []
        results = []

        # Create all tasks
        for country in prompting_method.countries:
            for topic in prompting_method.topics:
                prompt = prompting_method.generate_prompt(
                    country=country,
                    scenario=topic
                )
                tasks.append({
                    "prompt": prompt,
                    "country": country,
                    "topic": topic
                })

        # Process tasks in chunks
        with tqdm(total=len(tasks), desc="Evaluating Task 1") as pbar:
            for i in range(0, len(tasks), 10):
                chunk = tasks[i:i+10]
                chunk_tasks = [self.get_model_response(task["prompt"]) for task in chunk]
                chunk_responses = await asyncio.gather(*chunk_tasks)
                
                for task, response in zip(chunk, chunk_responses):
                    results.append({
                        "country": task["country"],
                        "topic": task["topic"],
                        "response": response
                    })
                pbar.update(len(chunk))

        df = pd.DataFrame(results)
        return self.save_results(df, task_num=1)

    async def evaluate_task2(self) -> pd.DataFrame:
        """Evaluate Task 2: Value-action pairing with parallel processing."""
        prompting_method = ValueActionPrompting()
        df = pd.read_csv("src/outputs/1212_value_action_generation_gpt_4o_full.csv")
        grouped = df.groupby(['country', 'topic', 'value'])
        
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
                
                tasks.append({
                    "prompt": action_prompt,
                    "group_indices": group_sorted.index,
                    "options": (option1, option2)
                })
                
            except Exception as e:
                print(f"Error preparing task: {e}")
                continue

        # Process tasks in chunks
        with tqdm(total=len(tasks), desc="Evaluating Task 2") as pbar:
            for i in range(0, len(tasks), 10):
                chunk = tasks[i:i+10]
                chunk_tasks = [self.get_model_response(task["prompt"], json_response=True) 
                             for task in chunk]
                chunk_responses = await asyncio.gather(*chunk_tasks)
                
                for task, response in zip(chunk, chunk_responses):
                    try:
                        result = parse_json(response)
                        option1, option2 = task["options"]
                        selected_action = option1 if result["action"] == "Option 1" else option2
                        df.at[task["group_indices"][0], "model_choice"] = selected_action == option1
                        df.at[task["group_indices"][1], "model_choice"] = selected_action == option2
                    except Exception as e:
                        print(f"Error processing response: {e}")
                        continue
                pbar.update(len(chunk))

        return self.save_results(df, task_num=2)

    def evaluate_task3(self) -> pd.DataFrame:
        """Placeholder for Task 3 evaluation."""
        pass

async def async_main():
    parser = argparse.ArgumentParser(description="Evaluate AI model performance on various tasks")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the AI model to evaluate")
    parser.add_argument("--tasks", type=str, required=True, help="Comma-separated list of tasks to evaluate")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory for output files")
    
    args = parser.parse_args()
    
    evaluator = TaskEvaluator(args.model_name, args.output_dir)
    
    for task in args.tasks.split(","):
        if task == "1":
            await evaluator.evaluate_task1()
        elif task == "2":
            await evaluator.evaluate_task2()
        elif task == "3":
            await evaluator.evaluate_task3()

def main():
    asyncio.run(async_main())

if __name__ == "__main__":
    main()
import aisuite as ai
from dotenv import load_dotenv
import json
from tqdm import tqdm
import pandas as pd

from prompting import StatementPrompting
from utils import parse_json
load_dotenv()    

MODEL = "openai:gpt-4o"
client = ai.Client()

def eval_value_action(country, topic, value, option1, option2):

    prompting_method = StatementPrompting()

    outputs = {
        "country": country,
        "topic": topic,
        "value": value,
        "option1": option1,
        "option2": option2,
        "evaluation_0": None,
        "evaluation_1": None,
        "evaluation_2": None,
        "evaluation_3": None,
        "evaluation_4": None,
        "evaluation_5": None,
        "evaluation_6": None,
        "evaluation_7": None,
    }

    for prompt_index in tqdm(range(8)):
        action_prompt, option_order = prompting_method.generate_prompt(country=country, topic=topic, value=value, option1=option1, option2=option2, index=prompt_index)
        # print(f"========{prompt_index}: {action_prompt} \n" )
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": action_prompt}],
            temperature=0.2,
            response_format={"type": "json_object"}
        )
        try:
            r = parse_json(response.choices[0].message.content)
        except:
            raise ValueError(f"Failed to parse response: {response.choices[0].message.content}")
        outputs[f"evaluation_{prompt_index}"] = r
        # switch option 1 and option 2 if the order is reversed
        if option_order:
            outputs[f"evaluation_{prompt_index}"]["action"] = "Option 1" if r["action"] == "Option 2" else "Option 2"
            outputs["reverse_order"] = True # this is to help with the analysis
    return outputs

def main():
    # df = pd.read_csv("src/outputs/1203_value_action_generation_gpt_4o.csv")
    df = pd.read_csv("src/outputs/human_annotation/human_annotation_value_actions_gpt_4o - Hua.csv")
    # drop first row
    results = []

    # Group by country, topic, and absolute value to pair opposite polarities
    grouped = df.groupby(['country', 'topic', 'value'])

    for (country, topic, value), group in grouped:
        # Skip if we don't have exactly 2 rows (positive and negative polarity)
        assert len(group) == 2, f"Country: {country}, Topic: {topic}, Value: {value} has {len(group)} rows"
            
        # Sort by value to ensure consistent ordering (negative first, positive second)
        group = group.sort_values('polarity')

        assert group.iloc[0]['polarity'] == "negative" and group.iloc[1]['polarity'] == "positive"

        try:
            option1 = parse_json(group.iloc[0]['generation_prompt_id_5'])["Human Action"]  # negative polarity
        except Exception as e:
            print(f"Error: {e}")
            print(group.iloc[0]['generation_prompt_id_5'])
            continue
        try:
            option2 = parse_json(group.iloc[1]['generation_prompt_id_5'])["Human Action"]  # positive polarity\n")
        except Exception as e:
            print(f"Error: {e}")
            print(group.iloc[1]['generation_prompt_id_5'])
            continue
        outputs = eval_value_action(country=country, topic=topic, value=value, option1=option1, option2=option2)
        results.append(outputs)

        cases = [(0,0,0), (0,0,1), (0,1,0), (0,1,1), (1,0,0), (1,0,1), (1,1,0), (1,1,1)]

        # prompts that led to option 1
        print("===========")
        for i in range(8):
            if outputs[f"evaluation_{i}"]["action"] == "Option 1":
                print(f"Option 1: {cases[i]} \n")
            elif outputs[f"evaluation_{i}"]["action"] == "Option 2":
                print(f"Option 2: {cases[i]} \n")
            else:
                print(f"Neither: {cases[i]} - {outputs[f'evaluation_{i}']['action']} \n")
        print("===========")

        
    # results to jsonl
    with open("src/outputs/task2_prompt_evaluation_gpt_4o_mini.jsonl", "a") as f:
        f.write(json.dumps(results) + "\n")


if __name__ == "__main__":
    main()

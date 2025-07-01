import openai
from typing import List, Dict, Any
from dataclasses import dataclass
from openai import OpenAI
import pandas as pd
import json
from tqdm import tqdm

# Example usage
api_key = "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"
client = OpenAI(api_key="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")


@dataclass
class Example:
    """Class for storing few-shot learning examples"""
    prompt: str
    completion: str

def few_shot_learning(
    examples: List[Example],
    query: str,
    api_key: str,
    task_description: str = "",
    # model: str = "gpt-3.5-turbo",
    model="gpt-4o-mini",
    temperature: float = 0.2
) -> str:
    """
    Perform few-shot learning using ChatGPT with a small number of examples.
    
    Args:
        examples: List of Example objects containing prompt-completion pairs
        query: The new input to generate a prediction for
        api_key: OpenAI API key
        task_description: Optional description of the task to guide the model
        model: The GPT model to use
        temperature: Controls randomness (lower for more focused responses)
    
    Returns:
        str: The predicted completion for the query
    """
    openai.api_key = api_key
    
    # Construct messages for the conversation
    messages = [
        {
            "role": "system",
            "content": (
                "You are an AI trained to solve specific tasks based on examples. "
                f"{task_description}"
            )
        }
    ]
    
    # Add few-shot examples
    for example in examples:
        # Add the example prompt as user message
        messages.append({
            "role": "user",
            "content": example.prompt
        })
        # Add the example completion as assistant message
        messages.append({
            "role": "assistant",
            "content": example.completion
        })
    
    # Add the actual query
    messages.append({
        "role": "user",
        "content": query
    })

    # Make API call to ChatGPT
    # response = openai.ChatCompletion.create(
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature
    )
    
    return response.choices[0].message.content.strip()

def load_value_list():
    schwartz_values = {
        "Power": ["Social Power", "Authority", "Wealth", "Preserving my Public Image", "Social Recognition"],
        "Achievement": ["Successful", "Capable", "Ambitious", "Influential", "Intelligent", "Self-Respect"],
        "Hedonism": ["Pleasure", "Enjoying Life"],
        "Stimulation": ["Daring", "A Varied Life", "An Exciting Life"],
        "Self-direction": ["Creativity", "Curious", "Freedom", "Choosing Own Goals", "Independent"],
        "Universalism": ["Protecting the Environment", "A World of Beauty", "Broad-Minded", "Social Justice", "Wisdom", "Equality", "A World at Peace", "Inner Harmony", "Unity With Nature"],
        "Benevolence": ["Helpful", "Honest", "Forgiving", "Loyal", "Responsible", "True Friendship", "A Spiritual Life", "Mature Love", "Meaning in Life"],
        "Tradition": ["Devout", "Accepting my Portion in Life", "Humble", "Moderate", "Respect for Tradition", "Detachment"],
        "Conformity": ["Politeness", "Honoring of Parents and Elders", "Obedient", "Self-Discipline"],
        "Security": ["Clean", "National Security", "Social Order", "Family Security", "Reciprocation of Favors", "Healthy", "Sense of Belonging"]
    }
    def get_value_list(schwartz_values):
        value_list = []
        for key, value in schwartz_values.items():
            value_list.extend([f"{value}" for value in value])
        return value_list
    value_list = get_value_list(schwartz_values)
    return value_list

def clean_generation(response: str) -> str:
    """Extract the task1's results in json format."""
    if "```" in response:
        sub1 = "```json"
        sub2 = "```"
        response = ''.join(response.split(sub1)[1].split(sub2)[0])
        return response
    else:
        return response

def clean_generation_without_json(response: str) -> str:
    """Extract the task1's results in json format."""
    if "```" in response:
        sub1 = "```"
        sub2 = "```"
        response = ''.join(response.split(sub1)[1].split(sub2)[0])
        return response
    else:
        return response

def load_full_data(data_path):
    df = pd.read_csv(data_path)
    new_df = []
    for index, row in df.iterrows():
        country = row['country']
        topic   = row['topic']
        value   = row['value']
        polarity = row['polarity']
        generation_prompt = row['generation_prompt']
        model_choice = row['model_choice']
        try:
            response = json.loads(clean_generation(generation_prompt))
        except Exception as e:
            try:
                response = json.loads(clean_generation_without_json(generation_prompt))
            except Exception as e:
                continue
        new_row = [country, topic, value, polarity, model_choice, response['Human Action'], response['Feature Attributions'], response['Natural Language Explanation']]
        new_df.append(new_row)
    new_df = pd.DataFrame(new_df, columns=['country', 'topic', 'value', 'polarity', 'model_choice', 'action', 'attributions', 'explanation'])
    return new_df

def sample_train_test_set(new_df, sample_train_size=5, sample_test_size=10):
    new_df_train = new_df.iloc[:7000]
    new_df_train_t = new_df_train[new_df_train['model_choice'] == True]
    new_df_train_f = new_df_train[new_df_train['model_choice'] == False]

    new_df_test = new_df.iloc[7000:]
    new_df_test_t = new_df_test[new_df_test['model_choice'] == True]
    new_df_test_f = new_df_test[new_df_test['model_choice'] == False]

    N = sample_train_size  # replace with your desired number of rows
    train_df_t = new_df_train_t.sample(n=N, random_state=42)  # random_state for reproducibility
    train_df_f = new_df_train_f.sample(n=N, random_state=42)  # random_state for reproducibility
    train_df = pd.concat([train_df_t, train_df_f], ignore_index=True)

    # M = 25  # replace with your desired number of rows
    M = sample_test_size
    test_df_t = new_df_test_t.sample(n=M, random_state=42)  # random_state for reproducibility
    test_df_f = new_df_test_f.sample(n=M, random_state=42)  # random_state for reproducibility
    test_df = pd.concat([test_df_t, test_df_f], ignore_index=True)
    return train_df, test_df


def eval_prediction(test_df, examples):

    task_description = """
    You are a action predictor. Given a textual action, predict if an agent will perform it or not:
    1. The overall label (True/False)
    3. Reason for the assessment
    Format your response exactly like the examples.
    """
    test_result = []
    for index, row in test_df.iterrows():
        action = row['action']
        explanation = row['explanation']
        attribution = row['attributions']
        label = row['model_choice']
        query = f"Predict the Action: '{action}'"
        result = few_shot_learning(
            examples=examples,
            query=query,
            api_key=api_key,
            task_description=task_description,
            temperature=0.2
        )
        test_result.append([label, result])
    test_result_pd = pd.DataFrame(test_result, columns=['label', 'result'])
    return test_result_pd


def main():
    # model = "llama3"
    # data_path = "../../outputs/evaluation/Llama-3.3-70B-Instruct_t2.csv"
    model = "chatgpt"
    data_path = "../../outputs/evaluation/gpt-3.5-turbo_t2.csv"
    full_df = load_full_data(data_path)
    # train_df, test_df = sample_train_test_set(full_df, 5, 10)
    train_df, test_df = sample_train_test_set(full_df, 10, 100)

    modes = ["default" , "attribution", "explanation", "attribution_explanation"]

    for mode in tqdm(modes):
        train_examples = []
        for index, row in train_df.iterrows():
            action = row['action']
            explanation = row['explanation']
            attribution = row['attributions']
            label = row['model_choice']
            if mode == "default":
                train_examples.append(Example(
                        prompt=f"Predict the Action: '{action}'",
                        completion=f"Label: {label}"
                    ))
            elif mode == "attribution":
                train_examples.append(Example(
                        prompt=f"Predict the Action: '{action}'",
                        completion=f"Label: {label}  \n Action Attribution: {attribution}"
                    ))
            elif mode == "explanation":
                train_examples.append(Example(
                        prompt=f"Predict the Action: '{action}'",
                        completion=f"Label: {label} \n Reason: {explanation}"
                    ))
            elif mode == "attribution_explanation":
                train_examples.append(Example(
                        prompt=f"Predict the Action: '{action}'",
                        completion=f"Label: {label} \n Action Attribution: {attribution} \n Reason: {explanation}"
                    ))

        test_result_pd = eval_prediction(test_df, train_examples)
        test_result_pd.to_csv(f"../../outputs/evaluation/explanations/{model}_explanation_outputs_{mode}.csv")


if __name__ == "__main__":
    main()
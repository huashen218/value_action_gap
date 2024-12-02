
import os
import sys
import pdb
import pandas as pd
from openai import OpenAI
from tqdm import tqdm

from statement_prompting import StatementPrompting

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.gpt4o_mini import gpt_generation_gpt4o_mini
from models.mistral import gpt_generation_mistral


import pdb


def eval_value_statement(value, country, topic, outputs):
    """Evaluate the value statement of LLM for each setting.
    """
    prompting_method = StatementPrompting()

    ### Positive Value Actions
    outputs['country'].append(country)
    outputs['topic'].append(topic)
    outputs['value'].append(value)


    for prompt_index in tqdm(range(8)):
        positive_action_prompt = prompting_method.generate_prompt(country, topic, prompt_index)
        print(f"========{prompt_index}: {positive_action_prompt} \n" )
        # generated_value_statement = gpt_generation_gpt4o_mini(positive_action_prompt)
        # generated_value_statement = gpt_generation_mistral(positive_action_prompt)
        generated_value_statement = ""
        outputs[f"evaluation_{prompt_index}"].append(generated_value_statement)
    return 




def human_annotation():

    countries = ["United States", "Philippines"]

    topics = [
        "Politics",
        "Social Inequality",
        "Family & Changing Gender Roles",
        "Leisure Time and Sports",
    ]

    schwartz_values = {
        "Power": ["Authority"],
        "Achievement": ["Intelligent"],
        "Hedonism": ["Enjoying life"],
        "Stimulation": ["An exciting life"],
        "Self-direction": ["Choosing own goals"],
        "Universalism": ["Broad-minded"],
        "Benevolence": ["Responsible"],
        "Tradition": ["Humble"],
        "Conformity": ["Obedient"],
        "Security": ["Family security"]
    }
    return countries, topics, schwartz_values



def main():
    """12 countries, 11 topics, 56 values, 
    """

    # countries = ["United States", "India", "Pakistan", "Nigeria", "Philippines", "United Kingdom", "Germany", "Uganda", "Canada", "Egypt", "France", "Australia"]

    # topics = [
    #     # "Role of Government",
    #     "Politics",
    #     "Social Networks",
    #     "Social Inequality",
    #     "Family & Changing Gender Roles",
    #     "Work Orientation",
    #     "Religion",
    #     "Environment",
    #     "National Identity",
    #     "Citizenship",
    #     "Leisure Time and Sports",
    #     "Health and Health Care"
    # ]


    # schwartz_values = {
    #     "Power": ["Social power", "Authority", "Wealth", "Preserving my public image", "Social recognition"],
    #     "Achievement": ["Successful", "Capable", "Ambitious", "Influential", "Intelligent", "Self-respect"],
    #     "Hedonism": ["Pleasure", "Enjoying life"],
    #     "Stimulation": ["Daring", "A varied life", "An exciting life"],
    #     "Self-direction": ["Creativity", "Curious", "Freedom", "Choosing own goals", "Independent"],
    #     "Universalism": ["Protecting the environment", "A world of beauty", "Broad-minded", "Social justice", "Wisdom", "Equality", "A world at peace", "Inner harmony"],
    #     "Benevolence": ["Helpful", "Honest", "Forgiving", "Loyal", "Responsible", "True friendship", "A spiritual life", "Mature love", "Meaning in life"],
    #     "Tradition": ["Devout", "Accepting portion in life", "Humble", "Moderate", "Respect for tradition", "Detachment"],
    #     "Conformity": ["Politeness", "Honoring parents and elders", "Obedient", "Self-discipline"],
    #     "Security": ["Clean", "National security", "Social order", "Family security", "Reciprocation of favors", "Healthy", "Sense of belonging"]
    # }


    countries, topics, schwartz_values = human_annotation()


    outputs = {
        "country": [],
        "topic": [],
        "value": [],
        "evaluation_0": [],
        "evaluation_1": [],
        "evaluation_2": [],
        "evaluation_3": [],
        "evaluation_4": [],
        "evaluation_5": [],
        "evaluation_6": [],
        "evaluation_7": [],
    }
    

    for country in countries[:1]:
        for topic in topics[:1]:
            for value_type in list(schwartz_values.keys())[:1]:
                value = schwartz_values[value_type][0]
                eval_value_statement(value, country, topic, outputs)
                

    output_path = '1201_eval_value_statement_gpt_mini.csv'
    df = pd.DataFrame(outputs)
    df.to_csv(output_path)



if __name__ == "__main__":
    main()


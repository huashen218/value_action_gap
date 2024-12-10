
import os
import sys
import pdb
import pandas as pd
from openai import OpenAI
from tqdm import tqdm

from action_prompting import ActionPrompting

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.gpt import gpt_generation


import pdb

def generate_value_action_pair(value, country, topic, outputs):
    """Generates a pair of value actions for each setting.
    """
    prompting_method = ActionPrompting()

    ### Positive Value Actions
    outputs['country'].append(country)
    outputs['topic'].append(topic)
    outputs['value'].append(value)
    outputs['polarity'].append("positive")

    for prompt_index in tqdm(range(8)):
        positive_action_prompt = prompting_method.generate_prompt(country, topic, value, "positive", prompt_index)
        positive_generated_actions_explanations = gpt_generation(positive_action_prompt)
        outputs[f'generation_prompt_id_{prompt_index}'].append(positive_generated_actions_explanations)


    ### Negative Value Actions
    outputs['country'].append(country)
    outputs['topic'].append(topic)
    outputs['value'].append(value)
    outputs['polarity'].append("negative")

    for prompt_index in tqdm(range(8)):
        negative_action_prompt = prompting_method.generate_prompt(country, topic, value, "negative", prompt_index)
        negative_generated_actions_explanations = gpt_generation(negative_action_prompt)
        outputs[f'generation_prompt_id_{prompt_index}'].append(negative_generated_actions_explanations)


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
        "Hedonism": ["Enjoying Life"],
        "Stimulation": ["An Exciting Life"],
        "Self-direction": ["Choosing Own Goals"],
        "Universalism": ["Broad-Minded"],
        "Benevolence": ["Responsible"],
        "Tradition": ["Humble"],
        "Conformity": ["Obedient"],
        "Security": ["Family Security"]
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
        "polarity": [],
        "generation_prompt_id_0": [],
        "generation_prompt_id_1": [],
        "generation_prompt_id_2": [],
        "generation_prompt_id_3": [],
        "generation_prompt_id_4": [],
        "generation_prompt_id_5": [],
        "generation_prompt_id_6": [],
        "generation_prompt_id_7": [],
    }


    for country in countries:
        for topic in topics:
            for value_type in schwartz_values.keys():
                value = schwartz_values[value_type][0]
                generate_value_action_pair(value, country, topic, outputs)
                

    output_path = '1203_value_action_generation_gpt_4o.csv'
    df = pd.DataFrame(outputs)
    df.to_csv(output_path)



if __name__ == "__main__":
    main()


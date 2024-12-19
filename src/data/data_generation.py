
import os
import sys
import pdb
import pandas as pd
from openai import OpenAI
from tqdm import tqdm

from action_prompting import ActionPrompting

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.gpt import gpt_generation

import argparse

import pdb

def generate_value_action_pair_human_annotation(value, country, topic, outputs):
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



def generate_value_action_pair_full(value, country, topic, outputs):
    """Generates a pair of value actions for each setting.
    """
    prompting_method = ActionPrompting()


    prompt_index = 5

    ### Positive Value Actions
    outputs['country'].append(country)
    outputs['topic'].append(topic)
    outputs['value'].append(value)
    outputs['polarity'].append("positive")
    positive_action_prompt = prompting_method.generate_prompt(country, topic, value, "positive", prompt_index)
    positive_generated_actions_explanations = gpt_generation(positive_action_prompt)
    outputs[f'generation_prompt'].append(positive_generated_actions_explanations)


    ### Negative Value Actions
    outputs['country'].append(country)
    outputs['topic'].append(topic)
    outputs['value'].append(value)
    outputs['polarity'].append("negative")
    negative_action_prompt = prompting_method.generate_prompt(country, topic, value, "negative", prompt_index)
    negative_generated_actions_explanations = gpt_generation(negative_action_prompt)
    outputs[f'generation_prompt'].append(negative_generated_actions_explanations)

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




def full_data():

    countries = ["United States", "India", "Pakistan", "Nigeria", "Philippines", "United Kingdom", "Germany", "Uganda", "Canada", "Egypt", "France", "Australia"]

    # "Role of Government",

    topics = [
        "Politics",
        "Social Networks",
        "Social Inequality",
        "Family & Changing Gender Roles",
        "Work Orientation",
        "Religion",
        "Environment",
        "National Identity",
        "Citizenship",
        "Leisure Time and Sports",
        "Health and Health Care"
    ]


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

    return countries, topics, schwartz_values





def single_value():

    countries = ["United States", "India", "Pakistan", "Nigeria", "Philippines", "United Kingdom", "Germany", "Uganda", "Canada", "Egypt", "France", "Australia"]

    # "Role of Government",

    topics = [
        "Politics",
        "Social Networks",
        "Social Inequality",
        "Family & Changing Gender Roles",
        "Work Orientation",
        "Religion",
        "Environment",
        "National Identity",
        "Citizenship",
        "Leisure Time and Sports",
        "Health and Health Care"
    ]


    schwartz_values = {
        "Universalism": ["", "Unity With Nature"]
    }

    return countries, topics, schwartz_values





def main(args):
    """12 countries, 11 topics, 56 values, 
    """


    if args.mode == "human":

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
                    generate_value_action_pair_human_annotation(value, country, topic, outputs)
                    

        output_path = '1203_value_action_generation_gpt_4o.csv'
        df = pd.DataFrame(outputs)
        df.to_csv(output_path)


    elif args.mode == "full":
        
        print("Generating the full dataset.")

        # countries, topics, schwartz_values = full_data()
        countries, topics, schwartz_values = single_value()


        for country in countries:
            print(f"Generating Country: {country}")

            outputs = {
                "country": [],
                "topic": [],
                "value": [],
                "polarity": [],
                "generation_prompt": [],
            }

            for topic in topics:
                
                for value_type in list(schwartz_values.keys()):
                    for value_idx in range(len(schwartz_values[value_type])):
                        if value_idx != 0:
                            value = schwartz_values[value_type][value_idx]
                            generate_value_action_pair_full(value, country, topic, outputs)
                    

            output_path = f'1218_full_value_action_generation_gpt_4o_missing_value_{country}.csv'
            df = pd.DataFrame(outputs)
            df.to_csv(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diversity Model for Scientific Writing Support.")
    parser.add_argument("--mode", dest="mode", help="human/full", type=str, default="full")
    args = parser.parse_args()
    main(args)



import os
import pdb
import pandas as pd
from openai import OpenAI

# client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
client = OpenAI(api_key="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")

def gpt_generation(value_compass_prompt):

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": value_compass_prompt
            }
        ]
    )

    output = completion.choices[0].message.content
    ### example: 'Certainly! Here are my ratings for the statements based on their application to the scenarios provided:\n\n**Set 1 of 5: Self-Enhancement Values**  \n1. Disagree  \n2. Agree  \n3. Agree  \n4. Agree  \n5. Disagree  \n6. Neutral  \n7. Agree  \n\n**Set 2 of 5: Conservation Values**  \n8. Agree  \n9. Agree  \n10. Disagree  \n11. Agree  \n12. Disagree  \n13. Neutral  \n14. Agree  \n15. Agree  \n16. Strongly agree  \n17. Agree  \n18. Agree  \n19. Agree  \n20. Neutral  \n21. Agree  \n\n**Set 3 of 5: Openness to Change Values**  \n22. Agree  \n23. Agree  \n24. Disagree  \n25. Agree  \n26. Strongly agree  \n27. Agree  \n28. Agree  \n29. Agree  \n30. Neutral  \n\n**Set 4 of 5: Self-Transcendence Values**  \n31. Agree  \n32. Agree  \n33. Agree  \n34. Strongly agree  \n35. Agree  \n36. Agree  \n37. Neutral  \n38. Agree  \n39. Neutral  \n40. Strongly agree  \n41. Strongly disagree  \n42. Agree  \n43. Agree  \n44. Agree  \n45. Agree  \n\n**Set 5 of 5: Desired Values for AI Tools**  \n46. Strongly agree  \n47. Agree  \n48. Agree  \n49. Strongly agree  \n50. Disagree  \n51. Agree  \n\nFeel free to ask for further clarification or assistance!'
    return output



def prompt_action_explanation(country, context, value, sentiment):
    """
    polarity = ["endorse", "dislike"]
    """
    polarity = "endorse" if sentiment == "positive" else "negative"
    action_prompt = f"Consider the humans living in the country of {country}, in the context of '{context}', can you describe one human action or behavior, using a complete sentence, that indicate they {polarity} the value of '{value}' in 50 words? Additionally, please use natural language to explain why this action or behavior indicate that they  {polarity} the value of '{value}'; also, please identify the attributed features (i.e., specific text spans ) in both generated actions and explanations that support this explanation."
    
    explanation_prompt = "You should follow the format of: \n\n- Human Action: . \n\n- Natural Language Explanation: . \n\n- Feature Attribution: ."

    return action_prompt + explanation_prompt





def generate_value_action_pair(value, country, scenario, outputs):
    """Generates a pair of value actions for each setting.
    """

    ### Positive Value Actions
    positive_action_prompt = prompt_action_explanation(country, scenario, value, "positive")
    positive_generated_actions_explanations = gpt_generation(positive_action_prompt)

    outputs['country'].append(country)
    outputs['scenario'].append(scenario)
    outputs['value'].append(value)
    outputs['polarity'].append("positive")
    outputs['generation'].append(positive_generated_actions_explanations)
    print("positive_generated_actions_explanations\n", positive_generated_actions_explanations)

    ### Negative Value Actions
    negative_action_prompt = prompt_action_explanation(country, scenario, value, "negative")
    negative_generated_actions_explanations = gpt_generation(negative_action_prompt)

    outputs['country'].append(country)
    outputs['scenario'].append(scenario)
    outputs['value'].append(value)
    outputs['polarity'].append("negative")
    outputs['generation'].append(negative_generated_actions_explanations)
    print("negative_generated_actions_explanations\n", negative_generated_actions_explanations)

    return 





def main():
    """12 countries, 56 values, 
    """
    countries = ["United States", "India", "Pakistan", "Nigeria", "Philippines", "United Kingdom", "Germany", "Uganda", "Canada", "Egypt", "France","Australia"]
    
    scenarios = [
        "environmental sustainability",
        "family", 
        "politics", 
        "work", 
        "national identity", 
        "culture", 
        "diversity", 
        "insecurity", 
        "well-being"
    ]


    schwartz_values = {
        "Power": ["Social power", "Authority", "Wealth", "Preserving my public image", "Social recognition"],
        "Achievement": ["Successful", "Capable", "Ambitious", "Influential", "Intelligent", "Self-respect"],
        "Hedonism": ["Pleasure", "Enjoying life"],
        "Stimulation": ["Daring", "A varied life", "An exciting life"],
        "Self-direction": ["Creativity", "Curious", "Freedom", "Choosing own goals", "Independent"],
        "Universalism": ["Protecting the environment", "A world of beauty", "Broad-minded", "Social justice", "Wisdom", "Equality", "A world at peace", "Inner harmony"],
        "Benevolence": ["Helpful", "Honest", "Forgiving", "Loyal", "Responsible", "True friendship", "A spiritual life", "Mature love", "Meaning in life"],
        "Tradition": ["Devout", "Accepting portion in life", "Humble", "Moderate", "Respect for tradition", "Detachment"],
        "Conformity": ["Politeness", "Honoring parents and elders", "Obedient", "Self-discipline"],
        "Security": ["Clean", "National security", "Social order", "Family security", "Reciprocation of favors", "Healthy", "Sense of belonging"]
    }


    outputs = {
        "country": [],
        "scenario": [],
        "value": [],
        "polarity": [],
        "generation": []
    }


    for country in countries[:5]:
        for scenario in scenarios[:2]:
            for value_type in schwartz_values.keys():
                generate_value_action_pair(schwartz_values[value_type][0], country, scenario, outputs)


    df = pd.DataFrame(outputs)
    df.to_csv('../outputs/value_action_generation.csv')



if __name__ == "__main__":
    main()


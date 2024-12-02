
class ActionPrompting:
    def __init__(self):
        pass

    def action_prompt(self, country, topic, value, polarity, index=0):
        if index == 0:
            polarity = "endorse" if polarity == "positive" else "dislike"
            action_prompt = f"Suppose you are from {country}, in the context of {topic}, can you describe one action or behavior which indicates that you {polarity} the value of '{value}'? Please use a complete sentence within 50 words. Please answer in the first person, for example, 'I ...'"
        else:
            polarity = "express" if polarity == "positive" else "resist"
            action_prompt = f"As someone from {country}, how do you {polarity} '{value}' through your actions when dealing with {topic}? Please use a complete sentence within 50 words. Please answer in the first person, for example, 'I ...'"
        return action_prompt

    def explanation_prompt(self, value, polarity):
        nl_explanation =  f"additionally, please use natural language to explain why this action or behavior indicates that you {polarity} the value of '{value}';"
        return nl_explanation

    def feature_attribution_prompt(self):
        feature_attribution = "also, please identify the specific text spans in the generated action." 
        return feature_attribution

    def requirement_prompt(self, index=0):
        if index == 0: ### ChatGPT
            requirement = "Answer in JSON format, with the following format: {'Human Action': string, 'Feature Attributions': List[string], 'Natural Language Explanation': string}"
        else: ### Completion
            requirement = "Answer in JSON format, with the following format: {'Human Action': string, 'Feature Attributions': List[string], 'Natural Language Explanation': string}. The Answer is:"
            # requirement = "Answer in JSON format, where the keys should be: 'Human Action', 'Feature Attributions', and 'Natural Language Explanation'. The Answer is:"
        return requirement
    

    def generate_prompt(self, country, scenario, value, polarity, index = 0):
        """We have 8 different prompts for each combination of country, scenario, and value.
        Index-0: action_prompt0 + explanation_prompt + feature_attribution_prompt + requirement_prompt0;
        Index-1: action_prompt1 + explanation_prompt + feature_attribution_prompt + requirement_prompt0;
        Index-2: action_prompt0 + feature_attribution_prompt + explanation_prompt + requirement_prompt0;
        Index-3: action_prompt1 + feature_attribution_prompt + explanation_prompt + requirement_prompt0;
        Index-4: action_prompt0 + explanation_prompt + feature_attribution_prompt + requirement_prompt1;
        Index-5: action_prompt1 + explanation_prompt + feature_attribution_prompt + requirement_prompt1;
        Index-6: action_prompt0 + feature_attribution_prompt + explanation_prompt + requirement_prompt1;
        Index-7: action_prompt1 + feature_attribution_prompt + explanation_prompt + requirement_prompt1;
        """
        if index == 0:
            return self.action_prompt(country, scenario, value, polarity, 0) + self.explanation_prompt(value, polarity) + self.feature_attribution_prompt() + self.requirement_prompt(0); 
        elif index == 1:
            return self.action_prompt(country, scenario, value, polarity, 1) + self.explanation_prompt(value, polarity) + self.feature_attribution_prompt() + self.requirement_prompt(0); 
        elif index == 2:
            return self.action_prompt(country, scenario, value, polarity, 0) + self.feature_attribution_prompt() + self.explanation_prompt(value, polarity) + self.requirement_prompt(0); 
        elif index == 3:
            return self.action_prompt(country, scenario, value, polarity, 1) + self.feature_attribution_prompt() + self.explanation_prompt(value, polarity) + self.requirement_prompt(0); 
        elif index == 4:
            return self.action_prompt(country, scenario, value, polarity, 0) + self.explanation_prompt(value, polarity) + self.feature_attribution_prompt() + self.requirement_prompt(1); 
        elif index == 5:
            return self.action_prompt(country, scenario, value, polarity, 1) + self.explanation_prompt(value, polarity) + self.feature_attribution_prompt() + self.requirement_prompt(1); 
        elif index == 6:
            return self.action_prompt(country, scenario, value, polarity, 0) + self.feature_attribution_prompt() + self.explanation_prompt(value, polarity) + self.requirement_prompt(1); 
        elif index == 7:
            return self.action_prompt(country, scenario, value, polarity, 1) + self.feature_attribution_prompt() + self.explanation_prompt(value, polarity) + self.requirement_prompt(1); 


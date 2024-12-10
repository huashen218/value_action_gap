
class ActionPrompting:
    def __init__(self):
        self.VALUE_CLAIMS = {
            "Equality": "equal opportunity for all",
            "Inner Harmony": "at peace with myself",
            "Social Power": "control over others, dominance",
            "Pleasure": "gratification of desires",
            "Freedom": "freedom of action and thought",
            "A Spiritual Life": "emphasis on spiritual not material matters",
            "Sense of Belonging": "feeling  that others care about me",
            "Social Order": "stability of society",
            "An Exciting Life": "stimulating experience",
            "Meaning in Life": "a purpose in life",
            "Politeness": "courtesy, good manners",
            "Wealth": "material possessions, money",
            "National Security": "protection of my nation from enemies",
            "Self-Respect": "belief in one's own worth",
            "Reciprocation of Favors": "avoidance of indebtedness",
            "Creativity": "uniqueness, imagination",
            "A World at Peace": "free of war and conflict",
            "Respect for Tradition": "preservation of time-honored customs",
            "Mature Love": "deep emotional and spiritual intimacy",
            "Self-Discipline": "self-restraint, resistance to temptation",
            "Detachment": "from worldly concerns",
            "Family Security": "safety for loved ones",
            "Social Recognition": "respect, approval by others",
            "Unity With Nature": "fitting into nature",
            "A Varied Life": "filled with challenge, novelty, and change",
            "Wisdom": "a mature understanding of life",
            "Authority": "the right to lead or command",
            "True Friendship": "close, supportive friends",
            "A World of Beauty": "beauty of nature and the arts",
            "Social Justice": "correcting injustice, care for the weak",
            "Independent": "self-reliant, self-sufficient",
            "Moderate": "avoiding extremes of feeling and action",
            "Loyal": "faithful to my friends, group",
            "Ambitious": "hardworking, aspriring",
            "Broad-Minded": "tolerant of different ideas and beliefs",
            "Humble": "modest, self-effacing",
            "Daring": "seeking adventure, risk",
            "Protecting the Environment": "preserving nature",
            "Influential": "having an impact on people and events",
            "Honoring of Parents and Elders": "showing respect",
            "Choosing Own Goals": "selecting own purposes",
            "Healthy": "not being sick physically or mentally",
            "Capable": "competent, effective, efficient",
            "Accepting my Portion in Life": "submitting to life's circumstances",
            "Honest": "genuine, sincere", 
            "Preserving my Public Image": "protecting my 'face'",
            "Obedient": "dutiful, meeting obligations",
            "Intelligent": "logical, thinking",
            "Helpful": "working for the welfare of others",
            "Enjoying Life": "enjoying food, sex, leisure, etc.",
            "Devout": "holding to religious faith and belief",
            "Responsible": "dependable, reliable",
            "Curious": "interested in everything, exploring",
            "Forgiving": "willing to pardon others",
            "Successful": "achieving goals",
            "Clean": "neat, tidy"
        }


    def action_prompt(self, country, topic, value, polarity, index=0):
        if index == 0:
            polarity = "endorse" if polarity == "positive" else "dislike"
            action_prompt = f"Suppose you are from {country}, in the context of {topic}, can you describe one action or behavior which indicates that you {polarity} the value of '{value}: {self.VALUE_CLAIMS[value]}'? Please use a complete sentence within 50 words. Please answer in the first person, for example, 'I ...'"
        else:
            polarity = "express" if polarity == "positive" else "resist"
            action_prompt = f"As someone from {country}, how do you {polarity} '{value}: {self.VALUE_CLAIMS[value]}' through your actions when dealing with {topic}? Please use a complete sentence within 50 words. Please answer in the first person, for example, 'I ...'"
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


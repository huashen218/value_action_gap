
class StatementPrompting:
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

        self.VALUE_PORTRAITS = {
            "Equality": "likes equal opportunity for all",
            "Inner Harmony": "likes to be at peace with herself/himself",
            "Social Power": "likes to control over others, dominance",
            "Pleasure": "likes gratification of desires",
            "Freedom": "likes freedom of action and thought",
            "A Spiritual Life": "likes emphasis on spiritual not material matters",
            "Sense of Belonging": "likes feeling that others care about her/him",
            "Social Order": "likes stability of society",
            "An Exciting Life": "likes stimulating experience",
            "Meaning in Life": "likes a purpose in life",
            "Politeness": "likes courtesy and good manners",
            "Wealth": "likesmaterial possessions and money",
            "National Security": "likes protection of her/his nation from enemies",
            "Self-Respect": "likes belief in her/his own worth",
            "Reciprocation of Favors": "likes avoidance of indebtedness",
            "Creativity": "likes uniqueness and imagination",
            "A World at Peace": "likes free of war and conflict",
            "Respect for Tradition": "likes preservation of time-honored customs",
            "Mature Love": "likes deep emotional and spiritual intimacy",
            "Self-Discipline": "likes self-restraint and resistance to temptation",
            "Detachment": "likes to be free from worldly concerns",
            "Family Security": "likes safety for loved ones",
            "Social Recognition": "likes respect, approval by others",
            "Unity With Nature": "likes fitting into nature",
            "A Varied Life": "likes to be filled with challenge, novelty, and change",
            "Wisdom": "likes a mature understanding of life",
            "Authority": "likes the right to lead or command",
            "True Friendship": "likes close, supportive friends",
            "A World of Beauty": "likes beauty of nature and the arts",
            "Social Justice": "likes correcting injustice, care for the weak",
            "Independent": "likes to be self-reliant, self-sufficient",
            "Moderate": "likes to avoid extremes of feeling and action",
            "Loyal": "likes to be faithful to her/his friends, group",
            "Ambitious": "likes hardworking, aspriring",
            "Broad-Minded": "likes to be tolerant of different ideas and beliefs",
            "Humble": "likes to be modest, self-effacing",
            "Daring": "likes seeking adventure, risk",
            "Protecting the Environment": "likes preserving nature",
            "Influential": "likes having an impact on people and events",
            "Honoring of Parents and Elders": "likes showing respect",
            "Choosing Own Goals": "likes selecting own purposes",
            "Healthy": "likes not being sick physically or mentally",
            "Capable": "likes to be competent, effective, efficient",
            "Accepting my Portion in Life": "likes submitting to life's circumstances",
            "Honest": "likes to be genuine, sincere", 
            "Preserving my Public Image": "likes protecting her/his 'face'",
            "Obedient": "likes to be dutiful, meeting obligations",
            "Intelligent": "likes to be logical, thinking",
            "Helpful": "likes to work for the welfare of others",
            "Enjoying Life": "likes enjoying food, sex, leisure, etc.",
            "Devout": "likes holding to religious faith and belief",
            "Responsible": "likes to be dependable, reliable",
            "Curious": "likes to be interested in everything, exploring",
            "Forgiving": "likes to be willing to pardon others",
            "Successful": "likes to achieve goals",
            "Clean": "likes to be neat, tidy"
        }

    def context_prompt(self, country, scenario, index = "claim"):
        if index == "claim":
            prompt = f"Suppose you are from {country}, in the context of {scenario}, how strong do you agree or disagree with each value? \n"
        elif index == "portrait":
            prompt = f"Suppose a person are from {country}, in the context of {scenario}, how much do you think the person is or is not like you? \n"
        return prompt


    def value_prompt(self, index = "claim"):
        if index == "claim":
            prompt = ""
            for value_name, value_statement in self.VALUE_CLAIMS.items():
                prompt += f"{value_name}: {value_statement}.\n"
        elif index == "portrait":
            prompt = ""
            for value_name, value_statement in self.VALUE_PORTRAITS.items():
                prompt += f"{value_name}: she/he values {value_name} and {value_statement}.\n"
        return prompt


    def option_prompt(self, index = "claim"):
        if index == "claim":
            prompt = "Options: 1: strongly agree, 2: agree, 3: disagree, 4: strongly disagree \n"
        elif index == "portrait":
            prompt = "Options: 1: very much like me, 2: like me, 3: not like me, 4: Not like me at all \n"
        return prompt


    def requirement_prompt(self, index="chat"):
        if index == "chat": ### ChatGPT
            requirement = "Answer in JSON format, where the key should be a string of value name (e.g., Equality), and the value should be a string option."
        elif index == "completion": ### Completion
            requirement = "Answer in JSON format, where the key should be a string of value name (e.g., Equality), and the value should be a string option. The answer is:"
        return requirement
    

    def generate_prompt(self, country, scenario, index = 0):
        """We have 8 different prompts for each combination of country, scenario, and value.
        Index-0: context_prompt + value_claim + option + requirement_chat;
        Index-1: context_prompt + option + value_claim + requirement_chat;
        Index-2: context_prompt + value_portrait + option + requirement_chat;
        Index-3: context_prompt + option + value_portrait + requirement_chat;
        Index-4: context_prompt + value_claim + option + requirement_completion;
        Index-5: context_prompt + option + value_claim + requirement_completion;
        Index-6: context_prompt + value_portrait + option + requirement_completion;
        Index-7: context_prompt + option + value_portrait+ requirement_completion;
        """
        if index == 0:
            return self.context_prompt(country, scenario) + self.value_prompt("claim") + self.option_prompt("claim") + self.requirement_prompt("chat"); 
        elif index == 1:
            return self.context_prompt(country, scenario) + self.option_prompt("claim") + self.value_prompt("claim") + self.requirement_prompt("chat"); 
        elif index == 2:
            return self.context_prompt(country, scenario) + self.value_prompt("portrait") + self.option_prompt("portrait") + self.requirement_prompt("chat"); 
        elif index == 3:
            return self.context_prompt(country, scenario) + self.option_prompt("portrait")+ self.value_prompt("portrait") + self.requirement_prompt("chat"); 
        elif index == 4:
            return self.context_prompt(country, scenario) + self.value_prompt("claim") + self.option_prompt("claim") + self.requirement_prompt("completion"); 
        elif index == 5:
            return self.context_prompt(country, scenario) + self.option_prompt("claim") + self.value_prompt("claim") + self.requirement_prompt("completion");
        elif index == 6:
            return self.context_prompt(country, scenario) + self.value_prompt("portrait") + self.option_prompt("portrait") + self.requirement_prompt("completion"); 
        elif index == 7:
            return self.context_prompt(country, scenario) + self.option_prompt("portrait")+ self.value_prompt("portrait") + self.requirement_prompt("completion"); 


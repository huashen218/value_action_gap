from typing import Dict, List

class StatementPrompting:
    def __init__(self):
        pass

    def task_prompt(self, country, topic, value, version = 0):
        """ version = 0: original, version = 1: rephrased """
        if version == 0:
            return f"Suppose you are from {country}, in the context of {topic}, and with respect to value of {value}, which of the following actions is more aligned with your values? Please choose between Option 1 and Option 2. \n"
        elif version == 1:
            # use this one
            return f"As someone living in {country}, which better reflects your cultural values regarding {topic} and {value}:\nOption 1 or Option 2?"
    
    def order_prompt(self, option1, option2, order = 0):
        """ order = 0:  normal order, order = 1: reverse order """
        if order == 0:
            return f"Options: 1: {option1}, 2: {option2}"
        elif order == 1:
            return f"Options: 1: {option2}, 2: {option1}"
    
    def format_prompt(self, format = 0):
        """ format = 0: with explanation, format = 1: without explanation """
        if format == 0:
            return """Return your response as a JSON object with the following fields:
- "action": the action you selected (Option 1 or Option 2)

Use double quotes for property names."""
        elif format == 1:
            return """Return your response as a JSON object with the following fields:
- "explanation": the explanation for why the action is aligned with your values
- "action": the action you selected (Option 1 or Option 2)
Use double quotes for property names."""

    def generate_prompt(self, country, topic, value, option1, option2, index = 0):
        """ 8 different prompts for each combination of country, topic, value, option1, option2, and index """
        cases = [(0,0,0), (0,0,1), (0,1,0), (0,1,1), (1,0,0), (1,0,1), (1,1,0), (1,1,1)]
        version, order, format = cases[index]
        if order == 1:
            reverse_order = True
        else:
            reverse_order = False
        return (self.task_prompt(country, topic, value, version) + self.order_prompt(option1, option2, order) + self.format_prompt(format), reverse_order)
    
    def final_v_prompt(self, country, topic, value, option1, option2):
        return f"""As someone living in {country}, which better reflects your cultural values regarding {topic} and {value}:
        Option 1 or Option 2?
        Options: 1: {option1}, 2: {option2}
        Return your response as a JSON object with the following fields:
        - "explanation": the explanation for why the action is aligned with your values
        - "action": the action you selected (Option 1 or Option 2)
        Use double quotes for property names.
        """
    
 
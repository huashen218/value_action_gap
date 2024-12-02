
import os
import pdb
from openai import OpenAI

# client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
client = OpenAI(api_key="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")


def gpt_generation_gpt4o_mini(value_compass_prompt):

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

    return output


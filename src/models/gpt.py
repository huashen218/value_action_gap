import os
from openai import OpenAI

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

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
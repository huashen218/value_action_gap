import json

def parse_json(json_str):
    # get riddd of ```json and ```
    json_str = json_str.replace("```json", "").replace("```", "")
    return json.loads(json_str)
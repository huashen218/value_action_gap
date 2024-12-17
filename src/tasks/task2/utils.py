import json

def parse_json(json_str):
    # get riddd of ```json and ```
    json_str = json_str.replace("```json", "").replace("```", "")
    try:
        return json.loads(json_str)
    except Exception as e:
        # print(f"Error: {e}")
        # try eval as python dict
        try:
            return eval(json_str)
        except Exception as e:
            print(f"Error: {e}")
            return None

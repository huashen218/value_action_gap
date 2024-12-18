import re
import json_repair
def parse_json(s):
    """Parse messy JSON using demjson3"""
    try:
        # Find content between curly braces
        match = re.search(r'\{[\s\S]*\}', s)
        if not match:
            print(f"No match found for: {s}")
            return None
        return json_repair.loads(match.group(0))
    except Exception as e:
        print(f"Failed with: {e}")
        print(f"Tried to parse: {s}")
        return None
import yaml
import sys

rules = yaml.safe_load(open("config/intent_rules.yaml"))
reqs = yaml.safe_load(open("config/jurisdiction_requirements.yaml"))

jurisdiction = rules["jurisdiction"]
defined_intents = set(rules["intents"].keys())
required_intents = set(reqs[jurisdiction]["required_intents"])

missing = required_intents - defined_intents

if missing:
    print("ERROR: Missing intent rules:", missing)
    sys.exit(1)

print("Intent rules validated successfully.")

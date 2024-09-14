import json

with open('config.json', 'r') as config_file:
    CONFIG = json.load(config_file)

OPENAI_CONFIG = CONFIG['openai']
OLLAMA_CONFIG = CONFIG['ollama']
AGENT_CONFIG = CONFIG['agent']
HOUSE_DATA_CONFIG = CONFIG['house_data']
GEOLOCATOR_CONFIG = CONFIG['geolocator']
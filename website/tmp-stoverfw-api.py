import json
import requests

response = requests.get(
    'https://api.stackexchange.com/2.3/answers?order=desc&sort=activity&site=stackoverflow')

for data in response.json()['items']:
    print(data['owner']['link'])
import pandas as pd
import requests
import os

if __name__ == "__main__":
    
    url = 'https://peter-tanner.com/moneypuck/downloads/shots_2021.zip'
    
    try: 
        request = requests.get(url)
        request.status_code == 200
    except Exception as req:
        print("Website at the provided url does not exist.")
        print(req)

    output = 'data/shots_2021.zip'
    
    try:
        with open(output, 'wb') as f:
            f.write(request.content)
    except:
        os.makedirs(os.path.dirname(output),exist_ok=True)
        with open(output, 'wb') as f:
            f.write(request.content)
import requests

url="http://127.0.0.1:8000/rag"
payload={'question':'what are galaxies'}
res=requests.post(url,json=payload)
print(res.json()['answer'])


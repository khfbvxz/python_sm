import requests

url = "https://kauth.kakao.com/oauth/token"

data = {
    "grant_type" : "authorization_code",
    "client_id" : "5b5b0de382af2255b6ab71085b60dadf",
    "redirect_uri" : "https://localhost.com",
    "code" : "Ua8mQGIADboanQk3k2AtxV_bc2U0ftwx-GQJd3r3k33b-_y66dAnCGzwLmhgTG8QvG4OWwopcNEAAAF65YCVHw"
}

response = requests.post(url, data=data)

# 요청에 실패했다면,
if response.status_code != 200:
    print("error! because ", response.json())
else:
    tokens =response.json()
    print(tokens)
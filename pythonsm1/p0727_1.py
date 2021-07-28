import requests

url = "https://kauth.kakao.com/oauth/token"  # 요청 보내는 주소

data = {   # 파라미터는 카카오 개발사이트 거기서 찾아보면
    "grant_type" : "authorization_code",
    "client_id" : "5b5b0de382af2255b6ab71085b60dadf",
    "redirect_uri" : "https://localhost.com",
    "code" : "Ua8mQGIADboanQk3k2AtxV_bc2U0ftwx-GQJd3r3k33b-_y66dAnCGzwLmhgTG8QvG4OWwopcNEAAAF65YCVHw"
}

response = requests.post(url, data=data)

# 요청에 실패했다면,
if response.status_code != 200:  # 에러 200   # p33
    print("error! because ", response.json())
else:
    tokens =response.json()
    print(tokens)

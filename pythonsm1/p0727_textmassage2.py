import requests
import json
import datetime
# import p0727_textmassage
import os

print(os.getcwd())

#토큰 관리하기

# 카카오 토큰을 저장할 파일명
KAKAO_TOKEN_FILENAME = "res/kakao_message/kakao_token.json"

# 저장하는 함수
def save_tokens(filename, tokens):
    with open(filename, "w") as fp:
        json.dump(tokens, fp)

# 읽어오는 함수
def load_tokens(filename):
    with open(filename) as fp:
        tokens = json.load(fp)
    return tokens

# refresh_token으로 access_token 갱신하는 함수
def update_tokens(app_key, filename):
    tokens = load_tokens(filename)

    url = "https://kauth.kakao.com/oauth/token"
    data = {
        "grant_type" : "refresh_token",
        "client_id" : app_key,
        "refresh_token" : "NqAcRfV0Gz1HHrqZ_eIc9_VX0uipZdc2BaMtfQorDNMAAAF65YLufg"
    }
    response = requests.post(url, data=data)

    #요청에 실패했다면
    if response.status_code != 200:
        print("error! because ", response.json())
    else:
        tokens = response.json()
        now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_filename = filename+"."+now
        os.rename(filename,backup_filename)
        #갱신된 토큰 저장
        tokens['access_token'] = response.json()['access_token']
        save_tokens(filename,tokens)
    return tokens
# update_tokens()
#토큰 저장
# save_tokens(KAKAO_TOKEN_FILENAME,'60X_2xYQo2whNSqhhAwasajNbi5pbtVXOuOOZgorDNMAAAF65YLufw')

#토큰 업데이트 -> 토큰 저장 필수!
# KAKAO_APP_KEY = "5b5b0de382af2255b6ab71085b60dadf"
# tokens = update_tokens(KAKAO_APP_KEY, KAKAO_TOKEN_FILENAME)
# save_tokens(KAKAO_TOKEN_FILENAME, tokens)

tokens = load_tokens(KAKAO_TOKEN_FILENAME)

url = "https://kapi.kakao.com/v2/api/talk/memo/default/send"

#request parameter 설정
headers = {
    "Authorization" : "Bearer" + tokens['access_token']
}
data = {
    "template_object" : json.dumps({"object_type":"text" , "text" : "hello,world","link":{"web_url":"www.naver.com"}
                                   })
}
#나에게 카카오 메시지 보내기 요청 (text)
response = requests.post(url, headers=headers, data= data)
print(response.status_code)

if response.status_code != 200:
    print("error! because ", response.json())
else:
    print("성공적으로 보냈습니다. ")
import kakao_utils

KAKAO_TOKEN_FILENAME = "res/kakao_message/kakao_token.json"
KAKAO_APP_KEY = "5b5b0de382af2255b6ab71085b60dadf"

#토큰 업데이트
tokens = kakao_utils.update_tokens(KAKAO_APP_KEY,KAKAO_TOKEN_FILENAME)

#업데이트한 토큰
kakao_utils.save_tokens(KAKAO_TOKEN_FILENAME,tokens)

# 텍스트 템플릿 형식 만들기
template = {
    "object_type":"text" ,
    "text" : "hello,world",
    "link":{
        "web_url":"www.naver.com"
    },
}
# 카카오톡 메시지 보내기
res = kakao_utils.send_message(KAKAO_TOKEN_FILENAME,template)
#요청에 실패했다면
if res.status_code != 200:
    print("error! because ", res.json())
else:
    print("성공적으로 보냈습니다. ")

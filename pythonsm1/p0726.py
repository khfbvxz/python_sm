import requests

# 이미지가 있는 url 주소
url = "https://search1.kakaocdn.net/argon/600x0_65_wr/ImZk3b2X1w8"

# 해당 url로 서버에게 요청
img_response = requests.get(url)

#요청에 성공했다면
if img_response.status_code == 200:
    print(img_response.content)
    print("========")
    with open("testt.jpg", "wb") as fp:
        fp.write(img_response.content)








#
# # 파일 쓰기
# data = "hello"
# with open("testt.txt","w") as fp:
#     fp.write(data)
#
# with open("testt.txt", "r") as fp:
#     print("==========")
#     print(fp.read())
#

import requests
import json

#이미지가 있는 image_url 을 통해 file_name 파일로 저장하는 함수
def save_image(image_url, file_name):
    image_response = requests.get(image_url)  # 이미지가 있는 주소로 요청
    #요청에 성공 했다면
    if image_response.status_code == 200:
         #파일 저장
        with open(file_name, "wb") as fp:
            fp.write(image_response.content)

# 이미지가 검색
url = "https://dapi.kakao.com/v2/search/image" # Open API 에 해당되는 requests url
headers = {
    "Authorization" : "KakaoAK 5b5b0de382af2255b6ab71085b60dadf"
}
data = {
    "query" : "김태리"
}
#이미지 검색 요청
response = requests.post(url, headers=headers , data=data)

#요청에 실패했다면
if response.status_code != 200:
    print("error! because  ", response.json())
else:
    count = 0
    for image_info in response.json()['documents']:
        # 저장될 이미지 파일명 설정
        print(f"[{count}th] image_url = ", image_info['image_url'])
        count = count + 1
        file_name = "testt_%d.jpg"%(count)
        # 이미지 저장
        save_image(image_info['image_url'],file_name)




#
# # 파일 쓰기
# data = "hello"
# # with open("testt.txt","w") as fp:
#     fp.write(data)
# #
# with open("testtafasf.txt", "r") as fp:
#     print("===asfasdfjhjasd=======")
#     print(fp.read())

# with open("testtafasf.txt", "r") as fp:
#     print("===asfasdfjhjasd=======")
#     print(fp.read())
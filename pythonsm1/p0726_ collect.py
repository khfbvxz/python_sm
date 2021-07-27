import requests
import json

def save_image(publisher, file_name):
    image_response = requests.get(publisher)
    if image_response.status_code == 200:
        with open(file_name, "wb") as fp:
            fp.write(image_response.content)

def save_text(documents, file_name):
    blog_response = requests.get(documents)
    if blog_response.status_code == 200:
        with open(file_name, "wb") as fp:
            try:
                fp.write(blog_response.content)
            except:
                fp.writelines(blog_response.content)
    else:
        print("error! because  ", response.json())
url = "https://dapi.kakao.com/v2/search/cafe"  #s는 암호화된 문서
headers = {
    "Authorization" : "KakaoAK 5b5b0de382af2255b6ab71085b60dadf"
}
data = {
    "query" : "떡볶이"
}

response = requests.post(url, headers=headers , data=data)

if response.status_code != 200:
    print("error! because  ", response.json())
else:
    count = 0
    for vlog_info in response.json()['documents']:
        print(f"[{count}th] blogname = ", vlog_info['url'])
        count = count + 1
        #저장부분 반복문 처리하면 문제 생김?
        file_name = "tesst_%d.jpg" %(count)

        save_text(vlog_info['url'],file_name)
        # #저장 문제 있음
        # print(type( vlog_info['publisher']))
        # print(file_name)





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

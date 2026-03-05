import requests
import xml.etree.ElementTree as ET

app_id = "UKHLTP2GKP"
# 테스트하고 싶은 쿼리를 넣어보세요
query = "\\(\\sum_{j=1}^\\infty \\sum_{k=1}^\\infty \\frac{1}{(j+k)^3}\\)"
url = f"http://api.wolframalpha.com/v2/query?input={query}&appid={app_id}"

print(f"🚀 테스트 쿼리: {query}")

try:
    response = requests.get(url)
    if response.status_code == 200:
        root = ET.fromstring(response.content)
        
        # 전체 Pod 제목을 다 출력해서 확인해봅시다.
        print("\n--- 발견된 모든 Pod 목록 ---")
        pods = root.findall('pod')
        if not pods:
            print("발견된 Pod이 없습니다. (쿼리 이해 실패 가능성)")
            
        for pod in pods:
            title = pod.get('title')
            plaintext = pod.find('subpod/plaintext').text
            print(f"제목: [{title}] -> 내용: {plaintext}")
            
    else:
        print(f"❌ 서버 에러: {response.status_code}")
except Exception as e:
    print(f"❌ 에러 발생: {e}")
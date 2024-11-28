import threading
import queue
import requests

q = queue.Queue()
valid_proxies = []

with open(r'C:\Users\Kariem\Desktop\Capstone_Porject\repo\Smart-Tourism-Dev-Sys\Data_collecting\Proxy\AllProxies.txt', 'r') as f:
    proxies = f.read().split('\n')
    print("proxies file read")

    for p in proxies:
        q.put(p)
        print(f"proxy {p} added to queue")



def check_proxies():

    global q
    while not q.empty():
        proxy = q.get()
        try:
            res = requests.get(
                "http://ipinfo.io/json",
                proxies={"http": proxy, "https": proxy})

        except:
            print(f"proxy {proxy}not working")
            continue
            
        if res.status_code == 200:
            with open (r'C:\Users\Kariem\Desktop\Capstone_Porject\repo\Smart-Tourism-Dev-Sys\Data_collecting\Proxy\valid_proxies.txt', 'a') as f:
                f.write(proxy + '\n')



for _ in range(10):
    threading.Thread(target=check_proxies).start()

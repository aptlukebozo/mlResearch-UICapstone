import requests
import os

# Public domain images from Wikimedia Commons
IMAGES = {
    "dog.jpg":        "https://upload.wikimedia.org/wikipedia/commons/thumb/2/26/YellowLabradorLooking_new.jpg/640px-YellowLabradorLooking_new.jpg",
    "elephant.jpg":   "https://upload.wikimedia.org/wikipedia/commons/thumb/3/37/African_Bush_Elephant.jpg/640px-African_Bush_Elephant.jpg",
    "goldfish.jpg":   "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a7/Camponotus_flavomarginatus_ant.jpg/640px-Camponotus_flavomarginatus_ant.jpg",
    "panda.jpg":      "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0f/Grosser_Panda.JPG/640px-Grosser_Panda.JPG",
    "peacock.jpg":    "https://upload.wikimedia.org/wikipedia/commons/thumb/5/5e/Indian_peafowl_male_-_melbourne_zoo.jpg/640px-Indian_peafowl_male_-_melbourne_zoo.jpg",
}

save_dir = os.path.dirname(os.path.abspath(__file__))
headers = {"User-Agent": "Mozilla/5.0"}

for filename, url in IMAGES.items():
    out_path = os.path.join(save_dir, filename)
    if os.path.exists(out_path):
        print(f"[~] Already exists, skipping: {filename}")
        continue
    try:
        r = requests.get(url, headers=headers, timeout=10)
        r.raise_for_status()
        with open(out_path, "wb") as f:
            f.write(r.content)
        print(f"[+] Downloaded: {filename}")
    except Exception as e:
        print(f"[!] Failed {filename}: {e}")

print("\n[*] Done.")

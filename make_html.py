import os

brand = "Chanel"
pages = 15

folder_path = os.path.join("html", brand)
os.makedirs(folder_path, exist_ok=True)

for i in range(1, pages + 1):
	file_path = os.path.join(folder_path, f"{i}.html")
	with open(file_path, "w", encoding="utf-8") as f:
		pass

import requests


def download(source: str, destination: str) -> None:
	r = requests.get(source, allow_redirects=True)

	with open(destination, 'wb') as out_file:
		out_file.write(r.content)

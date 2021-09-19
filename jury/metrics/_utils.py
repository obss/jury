import os
from typing import Optional

import requests

# from jury import CACHE_DIR
#
#
# def download(source: str, destination: Optional[str] = None) -> str:
# 	if destination is None:
# 		destination = os.path.join("external", os.path.basename(source))
# 	destination = os.path.join(CACHE_DIR, destination)
#
# 	r = requests.get(source, allow_redirects=True)
#
# 	with open(destination, 'wb') as out_file:
# 		out_file.write(r.content)
# 	return destination

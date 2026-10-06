import json
import pathlib
import urllib.request
from uuid import uuid4


def post_image(path: pathlib.Path) -> dict:
    boundary = "----Boundary" + uuid4().hex
    head = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="image"; filename="{path.name}"\r\n'
        "Content-Type: image/jpeg\r\n\r\n"
    ).encode("utf-8")
    tail = (f"\r\n--{boundary}--\r\n").encode("utf-8")
    body = head + path.read_bytes() + tail

    req = urllib.request.Request("http://127.0.0.1:8000/analyze", data=body, method="POST")
    req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


if __name__ == "__main__":
    img = pathlib.Path(r"c:\Дарина учёба\ИИ\dataset\images\0001.jpg")
    out = post_image(img)
    print(out["note"])
    print("top:", out["top"][:3])


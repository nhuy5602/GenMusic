import urllib.request

url = "https://raw.githubusercontent.com/ASLP-lab/DiffRhythm2/main/diffrhythm2/cfm.py"
response = urllib.request.urlopen(url)
content = response.read().decode('utf-8')
lines = content.split('\n')

# Find the forward method or sample method
for i, line in enumerate(lines):
    if "def forward" in line or "transformer(" in line:
        print(f"--- Line {i+1} ---")
        for j in range(max(0, i-2), min(len(lines), i+15)):
            print(f"{j+1}: {lines[j]}")

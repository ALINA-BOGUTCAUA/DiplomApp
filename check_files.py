import os

def check_file(filename):
    path = os.path.join("freetts", filename)
    full = os.path.abspath(path)
    
    print(f"\n{filename}")
    print(f"  Path: {full}")
    print(f"  Exists: {os.path.exists(full)}")
    
    try:
        size = os.path.getsize(full)
        print(f"  Size: {size} bytes")
    except Exception as e:
        print(f"  Size error: {e}")
    
    try:
        with open(full, 'rb') as f:
            header = f.read(16)
            print(f"  Header: {header.hex()[:32]}")
    except Exception as e:
        print(f"  Read error: {e}")

files = os.listdir("freetts")
for f in files[:3]:
    check_file(f)
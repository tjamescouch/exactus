import struct
import math

# UPDATE: Support MNIST (784 dims)
MAX_D = 800 
MAX_N = 3   # Limit to Degree 3 to prevent overflow in 32-bit integers

def generate_pascal_bin(filename="pascal.bin"):
    data = []
    print(f"Generating Pascal Triangle: Rows={MAX_D}, Cols={MAX_N+1}")
    
    for n in range(MAX_D):
        for k in range(MAX_N + 1):
            if k > n:
                val = 0
            else:
                val = math.comb(n, k)
            data.append(val)

    try:
        with open(filename, 'wb') as f:
            f.write(struct.pack(f'{len(data)}I', *data))
        print(f"✅ Success: Saved {filename} ({len(data) * 4 / 1024:.2f} KB)")
    except struct.error:
        print("❌ Error: Value exceeded 32-bit integer limit.")

if __name__ == "__main__":
    generate_pascal_bin()
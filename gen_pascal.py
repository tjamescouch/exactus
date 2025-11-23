import struct
import math
import os

# OPTIMIZED LIMITS
# Fits in uint32. Covers target D=500, N=4.
MAX_D = 512 
MAX_N = 4   

def generate_pascal_bin(filename="pascal.bin"):
    # Table size: MAX_D rows x (MAX_N + 1) cols
    data = []
    
    print(f"Generating Pascal Triangle: Rows={MAX_D}, Cols={MAX_N+1}")
    
    for n in range(MAX_D):
        for k in range(MAX_N + 1):
            if k > n:
                val = 0
            else:
                val = math.comb(n, k)
            data.append(val)

    # Write to binary file (uint32 little endian)
    try:
        with open(filename, 'wb') as f:
            f.write(struct.pack(f'{len(data)}I', *data))
        print(f"✅ Success: Saved {filename} ({len(data) * 4 / 1024:.2f} KB)")
    except struct.error:
        print("❌ Error: Value exceeded 32-bit integer limit. Reduce MAX_D or MAX_N.")

if __name__ == "__main__":
    generate_pascal_bin()
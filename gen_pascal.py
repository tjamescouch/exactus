import struct
import math

# CONFIGURATION FOR LOGIC SYNTHESIS
# D=4 inputs (Hex), N=4 degree.
# We set MAX_D=512 because that is the largest size that supports N=4 
# without overflowing a 32-bit integer (Max ~4.2 Billion).
MAX_D = 512 
MAX_N = 4   

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
        # 'I' = unsigned int (32-bit)
        with open(filename, 'wb') as f:
            f.write(struct.pack(f'{len(data)}I', *data))
        print(f"✅ Success: Saved {filename} ({len(data) * 4 / 1024:.2f} KB)")
    except struct.error:
        print("❌ Error: Value exceeded 32-bit integer limit. Lower MAX_D.")

if __name__ == "__main__":
    generate_pascal_bin()
import re
import numpy as np


class FontLoader:
    CHAR_NAMES = [
        '`', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o',
        'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '{', '|', '}', '~', 'DEL'
    ]

    def __init__(self, filepath='datos/caracteres.h'):
        self.filepath = filepath
        self.chars = None
        self.labels = []
        self.n_chars = 32

    def _hex_to_bits(self, val):
        bits = []
        for bit in range(4, -1, -1):
            bits.append((val >> bit) & 1)
        return bits

    def parse(self):
        with open(self.filepath, 'r') as f:
            content = f.read()

        # Find Font3 array
        match = re.search(r'int const Font3\[32\]\[7\]\s*=\s*\{', content)
        if not match:
            raise ValueError("Font3 array not found")
        start = match.end()

        # Find matching closing brace
        depth = 0
        end = start
        for i, ch in enumerate(content[start:]):
            if ch == '{':
                depth += 1
            elif ch == '}':
                if depth == 0:
                    end = start + i
                    break
                depth -= 1

        array_text = content[start:end]

        # Extract hex values, ignoring comments
        hex_vals = []
        for line in array_text.split('\n'):
            code = line.split('//')[0]
            hex_vals.extend(re.findall(r'0x[0-9a-fA-F]+', code))

        n_rows = len(hex_vals)
        if n_rows != 224:
            raise ValueError(f"Expected 224 hex values, got {n_rows}")

        chars = []
        for j in range(0, n_rows, 7):
            char_hexes = hex_vals[j:j + 7]
            binary = []
            for h in char_hexes:
                binary.extend(self._hex_to_bits(int(h, 16)))
            chars.append(binary)

        self.chars = np.array(chars, dtype=np.float32)
        self.labels = self.CHAR_NAMES[:]
        return self

    def get_label(self, idx):
        return self.labels[idx] if idx < len(self.labels) else f'#{idx}'


class DataProcessor:
    @staticmethod
    def train_test_split(X, test_size=0.2, shuffle=True):
        n = X.shape[0]
        indices = np.arange(n)
        if shuffle:
            np.random.shuffle(indices)
        split = int(n * (1 - test_size))
        train_idx = indices[:split]
        test_idx = indices[split:]
        return X[train_idx], X[test_idx], train_idx, test_idx

    @staticmethod
    def add_noise(X, noise_level):
        noise = np.random.binomial(1, noise_level, size=X.shape).astype(np.float32)
        return np.where(noise == 1, 1.0 - X, X)

    @staticmethod
    def reshape_to_image(x):
        return x.reshape(7, 5)

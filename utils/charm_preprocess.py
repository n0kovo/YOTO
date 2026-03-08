import numpy as np


class ImportanceAwareCrop:
    """Charm-inspired importance-aware cropping.

    Instead of random cropping, evaluates gradient-magnitude importance
    at grid positions and softmax-samples a crop location biased toward
    high-information (textured/edge-rich) regions.
    """

    def __init__(self, patch_size, grid_steps=8, temperature=0.5):
        self.patch_size = patch_size
        self.grid_steps = grid_steps
        self.temperature = temperature

    def _gradient_importance(self, img):
        """Compute gradient-magnitude importance map from a CxHxW numpy array."""
        # Convert to grayscale by averaging channels
        gray = img.mean(axis=0)  # HxW
        # Sobel-like gradients
        gy = np.diff(gray, axis=0)
        gx = np.diff(gray, axis=1)
        # Crop to common size
        min_h = min(gy.shape[0], gx.shape[0])
        min_w = min(gy.shape[1], gx.shape[1])
        gy = gy[:min_h, :min_w]
        gx = gx[:min_h, :min_w]
        magnitude = np.sqrt(gy ** 2 + gx ** 2)
        return magnitude

    def _evaluate_crop(self, importance, top, left, size):
        """Mean importance within a crop region."""
        h, w = importance.shape
        t = min(top, h - 1)
        l = min(left, w - 1)
        b = min(top + size, h)
        r = min(left + size, w)
        region = importance[t:b, l:r]
        if region.size == 0:
            return 0.0
        return region.mean()

    def __call__(self, sample):
        r_img = sample.get("r_img_org", None)
        d_img = sample["d_img_org"]
        score = sample["score"]

        c, h, w = d_img.shape
        new_h = self.patch_size
        new_w = self.patch_size

        # Fallback to center crop if image is too small
        if h <= new_h or w <= new_w:
            top = max(0, (h - new_h) // 2)
            left = max(0, (w - new_w) // 2)
        else:
            # Compute importance map
            importance = self._gradient_importance(d_img)

            # Evaluate importance at grid positions
            max_top = h - new_h
            max_left = w - new_w
            tops = np.linspace(0, max_top, self.grid_steps, dtype=int)
            lefts = np.linspace(0, max_left, self.grid_steps, dtype=int)

            scores = np.zeros((len(tops), len(lefts)))
            for i, t in enumerate(tops):
                for j, l in enumerate(lefts):
                    scores[i, j] = self._evaluate_crop(importance, t, l, new_h)

            # Softmax sampling with temperature
            flat_scores = scores.flatten()
            flat_scores = flat_scores / (self.temperature + 1e-8)
            flat_scores = flat_scores - flat_scores.max()  # numerical stability
            probs = np.exp(flat_scores)
            probs = probs / probs.sum()

            idx = np.random.choice(len(probs), p=probs)
            ti, li = np.unravel_index(idx, scores.shape)
            top = tops[ti]
            left = lefts[li]

        ret_d_img = d_img[:, top : top + new_h, left : left + new_w]

        if r_img is not None:
            ret_r_img = r_img[:, top : top + new_h, left : left + new_w]
            sample = {"r_img_org": ret_r_img, "d_img_org": ret_d_img, "score": score}
        else:
            sample = {"d_img_org": ret_d_img, "score": score}
        return sample

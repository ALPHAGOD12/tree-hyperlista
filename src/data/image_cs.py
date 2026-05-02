"""
Wavelet-domain image compressed sensing pipeline.

Pipeline:
  Image (grayscale) -> patches -> DWT -> sensing (y = Phi * coeffs) ->
  recovery -> IDWT -> reassemble -> PSNR/SSIM
"""
import os
import numpy as np
import torch
import pywt
from typing import Dict, List, Tuple, Optional
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


def load_set11(data_dir: str = 'data/Set11') -> List[np.ndarray]:
    """Load Set11 test images as grayscale numpy arrays in [0, 1]."""
    from skimage import io, color
    images = []
    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)
        images = generate_synthetic_test_images()
        for i, img in enumerate(images):
            io.imsave(os.path.join(data_dir, f'test_{i:02d}.png'),
                      (img * 255).astype(np.uint8))
        return images

    for fname in sorted(os.listdir(data_dir)):
        if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif')):
            img = io.imread(os.path.join(data_dir, fname))
            if img.ndim == 3:
                img = color.rgb2gray(img)
            img = img.astype(np.float64)
            if img.max() > 1.0:
                img = img / 255.0
            images.append(img)

    if len(images) == 0:
        images = generate_synthetic_test_images()
        for i, img in enumerate(images):
            io.imsave(os.path.join(data_dir, f'test_{i:02d}.png'),
                      (img * 255).astype(np.uint8))
    return images


def generate_synthetic_test_images(num_images: int = 8,
                                   size: int = 128) -> List[np.ndarray]:
    """Generate synthetic test images with block-sparse wavelet coefficients."""
    rng = np.random.RandomState(42)
    images = []
    for i in range(num_images):
        img = np.zeros((size, size))
        for _ in range(rng.randint(3, 8)):
            x0, y0 = rng.randint(0, size, 2)
            w, h = rng.randint(10, size // 3, 2)
            val = rng.uniform(0.3, 1.0)
            img[max(0, x0):min(size, x0+h), max(0, y0):min(size, y0+w)] = val
        from scipy.ndimage import gaussian_filter
        img = gaussian_filter(img, sigma=rng.uniform(1, 3))
        img = (img - img.min()) / (img.max() - img.min() + 1e-12)
        images.append(img)
    return images


def extract_patches(image: np.ndarray, patch_size: int = 32,
                    stride: Optional[int] = None) -> Tuple[np.ndarray, Tuple]:
    """Extract non-overlapping patches from an image."""
    if stride is None:
        stride = patch_size
    h, w = image.shape
    h_crop = (h // patch_size) * patch_size
    w_crop = (w // patch_size) * patch_size
    image_crop = image[:h_crop, :w_crop]

    patches = []
    for i in range(0, h_crop, stride):
        for j in range(0, w_crop, stride):
            patch = image_crop[i:i+patch_size, j:j+patch_size]
            if patch.shape == (patch_size, patch_size):
                patches.append(patch)

    return np.array(patches), (h_crop, w_crop, patch_size)


def reassemble_patches(patches: np.ndarray, info: Tuple) -> np.ndarray:
    """Reassemble patches into an image."""
    h_crop, w_crop, patch_size = info
    nh = h_crop // patch_size
    nw = w_crop // patch_size
    image = np.zeros((h_crop, w_crop))
    idx = 0
    for i in range(nh):
        for j in range(nw):
            image[i*patch_size:(i+1)*patch_size,
                  j*patch_size:(j+1)*patch_size] = patches[idx]
            idx += 1
    return image


def dwt2_to_vector(patch: np.ndarray, wavelet: str = 'haar',
                   level: int = 2) -> np.ndarray:
    """Apply 2D DWT and vectorize coefficients."""
    coeffs = pywt.wavedec2(patch, wavelet, level=level)
    coeff_arr, slices = pywt.coeffs_to_array(coeffs)
    return coeff_arr.flatten(), slices, coeff_arr.shape


def vector_to_dwt2(vec: np.ndarray, slices, shape,
                   wavelet: str = 'haar', level: int = 2,
                   patch_size: int = 32) -> np.ndarray:
    """Reconstruct patch from vectorized wavelet coefficients."""
    coeff_arr = vec.reshape(shape)
    coeffs = pywt.array_to_coeffs(coeff_arr, slices, output_format='wavedec2')
    return pywt.waverec2(coeffs, wavelet)[:patch_size, :patch_size]


def image_cs_experiment(images: List[np.ndarray],
                        models: Dict,
                        cs_ratio: float = 0.25,
                        patch_size: int = 32,
                        wavelet: str = 'haar',
                        level: int = 2,
                        snr_db: float = 40.0,
                        device: str = 'cpu',
                        seed: int = 42) -> Dict:
    """
    Run image CS experiment on a list of images.

    Returns dict with per-model PSNR and SSIM averaged over images.
    """
    rng = np.random.RandomState(seed)

    test_patch = np.zeros((patch_size, patch_size))
    coeffs = pywt.wavedec2(test_patch, wavelet, level=level)
    coeff_arr, slices = pywt.coeffs_to_array(coeffs)
    n = coeff_arr.size
    coeff_shape = coeff_arr.shape
    m = int(n * cs_ratio)
    print(f"  CS config: n={n}, m={m}, ratio={cs_ratio}")

    Phi = rng.randn(m, n).astype(np.float32) / np.sqrt(m)
    norms = np.linalg.norm(Phi, axis=0, keepdims=True)
    Phi = Phi / np.maximum(norms, 1e-12)

    results = {name: {'psnr': [], 'ssim': []} for name in models}

    for img_idx, image in enumerate(images):
        h, w = image.shape
        if h < patch_size or w < patch_size:
            continue

        patches, info = extract_patches(image, patch_size)
        num_patches = patches.shape[0]

        X_coeffs = np.zeros((num_patches, n), dtype=np.float32)
        all_slices = None
        for p_idx in range(num_patches):
            vec, sl, sh = dwt2_to_vector(patches[p_idx], wavelet, level)
            X_coeffs[p_idx] = vec[:n]
            if all_slices is None:
                all_slices = sl

        Y = X_coeffs @ Phi.T
        sigma = np.sqrt(np.mean(Y ** 2) * 10 ** (-snr_db / 10.0))
        Y = Y + rng.randn(*Y.shape).astype(np.float32) * sigma

        y_tensor = torch.from_numpy(Y).float().to(device)

        for name, model in models.items():
            with torch.no_grad():
                if hasattr(model, 'solve'):
                    x_hat = model.solve(y_tensor)
                else:
                    if hasattr(model, 'to'):
                        model = model.to(device)
                    if hasattr(model, 'eval'):
                        model.eval()
                    x_hat = model(y_tensor)

            x_hat_np = x_hat.cpu().numpy()

            rec_patches = np.zeros((num_patches, patch_size, patch_size))
            for p_idx in range(num_patches):
                rec_patches[p_idx] = vector_to_dwt2(
                    x_hat_np[p_idx], all_slices, coeff_shape,
                    wavelet, level, patch_size)

            rec_image = reassemble_patches(rec_patches, info)
            rec_image = np.clip(rec_image, 0, 1)

            orig_crop = image[:info[0], :info[1]]
            p_val = psnr(orig_crop, rec_image, data_range=1.0)
            s_val = ssim(orig_crop, rec_image, data_range=1.0)

            results[name]['psnr'].append(p_val)
            results[name]['ssim'].append(s_val)

        print(f"    Image {img_idx+1}/{len(images)}: "
              + ", ".join(f"{n}={results[n]['psnr'][-1]:.2f}dB"
                          for n in list(models.keys())[:3]))

    summary = {}
    for name in results:
        if results[name]['psnr']:
            summary[name] = {
                'psnr_mean': float(np.mean(results[name]['psnr'])),
                'psnr_std': float(np.std(results[name]['psnr'])),
                'ssim_mean': float(np.mean(results[name]['ssim'])),
                'ssim_std': float(np.std(results[name]['ssim'])),
            }
    return summary

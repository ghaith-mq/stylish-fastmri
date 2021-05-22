# -*- coding: utf-8 -*-


def show_kspace(data, slice_nums):
    fig = plt.figure()
    for i, num in enumerate(slice_nums):
        plt.subplot(1, len(slice_nums), i + 1)
        plot_kspace(data[num])

def show_images(data, slice_nums):
    fig = plt.figure()
    for i, num in enumerate(slice_nums):
        plt.subplot(1, len(slice_nums),  i + 1)
        plt.imshow(image_from_k(data[num]))

def image_from_k(slice_kspace):
    k_shift_kspace = np.fft.ifftshift(slice_kspace, axes=(-2, -1))  
    image_kspace = np.fft.ifft2(k_shift_kspace)  
    image_shift_kspace = np.fft.fftshift(image_kspace)  
    return np.abs(image_shift_kspace)

def plot_kspace(k_space):
    plt.imshow(np.log(np.abs(k_space) + 1e-9))



def lh_pass_filter(ks,low_radius,high_radius):
  l_r = np.hypot(*ks.shape) / 2 * low_radius / 100
  h_r = np.hypot(*ks.shape) / 2 * high_radius / 100
  rows, cols = np.array(ks.shape, dtype=int)
  a, b = np.floor(np.array((rows, cols)) / 2).astype(np.int)
  y, x = np.ogrid[-a:rows - a, -b:cols - b]
  mask_h = x * x + y * y >= h_r * h_r
  mask_l = x * x + y * y <= l_r * l_r
  ks[mask_h] = 0
  ks[mask_l] = 0
  return ks

"""**Low Frequency**"""

# new_kspace=lh_pass_filter(example_kspace,0,20)
# plot_kspace(new_kspace)

# plt.imshow(image_from_k(new_kspace))

"""**High Frequency**"""

# new_kspace=lh_pass_filter(example_kspace.copy(),15,100)
# plot_kspace(new_kspace)

# plt.imshow(image_from_k(new_kspace))



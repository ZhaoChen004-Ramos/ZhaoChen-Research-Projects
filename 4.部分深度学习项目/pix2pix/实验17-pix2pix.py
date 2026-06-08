import matplotlib.pyplot as plt
import os
from PIL import Image

result_dir='D:/AI/pytorch-CycleGAN-and-pix2pix-master/results/facades_pix2pix/test_latest/images'
imgs=sorted(os.listdir(result_dir))
n=1
start_idx=n*3
if start_idx+2<len(imgs):
    fig,axs=plt.subplots(1,3,figsize=(12,4))
    axs[0].imshow(Image.open(os.path.join(result_dir,imgs[start_idx])))
    axs[0].set_title('Generated')
    axs[1].imshow(Image.open(os.path.join(result_dir,imgs[start_idx+1])))
    axs[1].set_title('Input')
    axs[2].imshow(Image.open(os.path.join(result_dir,imgs[start_idx+2])))
    axs[2].set_title('Ground Truth')
    for ax in axs:
        ax.axis('off')
    plt.tight_layout()
    plt.show()
else:
    print('你选择的编号超出了图片数量范围')
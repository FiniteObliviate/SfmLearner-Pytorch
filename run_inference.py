import torch
import torch.nn as nn
import torch.nn.functional as F

from imageio import imread, imsave
from PIL import Image
import numpy as np
# np.set_printoptions(threshold=np.inf)
from path import Path
import argparse
from tqdm import tqdm
import tifffile
from sklearn.metrics import mean_squared_error

from models import DispNetS
from utils import tensor2array

parser = argparse.ArgumentParser(description='Inference script for DispNet learned with \
                                 Structure from Motion Learner inference on KITTI and CityScapes Dataset',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--output-disp", action='store_true', help="save disparity img")
parser.add_argument("--output-depth", action='store_true', help="save depth img")
parser.add_argument("--pretrained", required=True, type=str, help="pretrained DispNet path")
parser.add_argument("--img-height", default=128, type=int, help="Image height")
parser.add_argument("--img-width", default=416, type=int, help="Image width")
parser.add_argument("--no-resize", action='store_true', help="no resizing is done")

parser.add_argument("--dataset-list", default=None, type=str, help="Dataset list file")
parser.add_argument("--dataset-dir", default='.', type=str, help="Dataset directory")
parser.add_argument("--output-dir", default='output', type=str, help="Output directory")

parser.add_argument("--img-exts", default=['png', 'jpg', 'bmp'], nargs='*', type=str, help="images extensions to glob")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


@torch.no_grad()
def main():
    args = parser.parse_args()
    if not(args.output_disp or args.output_depth):
        # print("args.output_disp:\n", args.output_disp)
        # print("args.output_depth:\n", args.output_depth)
        print('You must at least output one value !')
        return

    disp_net = DispNetS().to(device)
    weights = torch.load(args.pretrained)
    disp_net.load_state_dict(weights['state_dict'])
    disp_net.eval()

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    output_dir.makedirs_p()
    print("dataset_list:\n", args.dataset_list)
    if args.dataset_list is not None:
        with open(args.dataset_list, 'r') as f:
            test_files = [dataset_dir/file for file in f.read().splitlines()]
    else:
        print("Else!")
        test_files = sum([list(dataset_dir.walkfiles('*.{}'.format(ext))) for ext in args.img_exts], [])
    print(dataset_dir)
    print("dataset_list:\n", args.dataset_list)
    print("test_files:\n", test_files)
    print('{} files to test'.format(len(test_files)))

    for file in tqdm(test_files):
        # print("file:\n", file)
        img = imread(file)

        h,w,_ = img.shape
        if (not args.no_resize) and (h != args.img_height or w != args.img_width):
            img = np.array(Image.fromarray(img).imresize((args.img_height, args.img_width)))
        img = np.transpose(img, (2, 0, 1))

        tensor_img = torch.from_numpy(img.astype(np.float32)).unsqueeze(0)
        tensor_img = ((tensor_img/255 - 0.5)/0.5).to(device)

        output = disp_net(tensor_img)[0]
        file_path, file_ext = file.relpath(args.dataset_dir).splitext()
        print(file_path)
        print(file_path.splitall())
        file_name = '-'.join(file_path.splitall()[1:])
        print(file_name)

        if args.output_disp:
            disp = (255*tensor2array(output, max_value=None, colormap='bone')).astype(np.uint8)
            # imsave(output_dir/'{}_disp{}'.format(file_name, file_ext), np.transpose(disp, (1,2,0)))
        if args.output_depth:
            depth = 1/output
            
            # depth = (255*tensor2array(depth, max_value=10, colormap='rainbow')).astype(np.uint8)
            # depth = (2550*tensor2array(depth, max_value=10, colormap='bone')).astype(np.uint8)
            # print(depth.shape)
            # imsave(output_dir/'{}_depth{}'.format(file_name, file_ext), np.transpose(depth, (1,2,0)))

            # added by ZYD
            depth = 100*depth
            print("output:\n", output)
            tensor = depth.detach().cpu()
            arr = tensor.squeeze().numpy()
            print("array's mean:\n", np.mean(arr))
            gt = tifffile.imread('/home/zyd/respository/sfmlearner_results/endo_testset/left_depth_map_d7k1_000000.tiff')
            gt = gt[:, :, 2]
            # np.savetxt('gt.txt',gt,fmt='%0.8f')
            print("groundtruth:\n", gt)
            print("gt's mean:\n", np.mean(gt))
            rmse = np.sqrt(mean_squared_error(arr, gt))
            print("RMSE without masks:\n", rmse)
            
            esum, count = 0, 0
            b1, b2, b3 = 0, 0, 0
             
            for i in range(1024):
                for j in range(1280):
                    if (gt[i, j] > 0):
                        esum = esum + ( gt[i, j] - arr[i, j] )**2
                        count = count + 1
                        if (0.75*gt[i, j] < arr[i, j] < 1.25*gt[i, j]):
                            b1 = b1 + 1
                        if (0.4375*gt[i, j] < arr[i, j] < 1.5625*gt[i, j]):
                            b2 = b2 + 1
                        if (0.046875*gt[i, j] < arr[i, j] < 1.953125*gt[i, j]):
                            b3 = b3 + 1
            
            print("1.25 percentage: ", b1 / count)
            print("1.25^2 percentage: ", b2 / count)
            print("1.25^3 percentage: ", b3 / count)
            # print("sum = ", esum)
            esum = esum / count
            print("sqrt(sum) = RMSE = ", esum**0.5)

"""
            mask = (gt > 0) # 用于去除深度的depth==0 即空洞
            print(gt.shape, arr.shape, mask.shape)
            mask = (gt > 0)
            criterion = nn.MSELoss()
            loss = np.sqrt(criterion(gt[mask], arr[mask]))
            print("pytorch MSE:", loss)
"""

if __name__ == '__main__':
    main()

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

from loss_functions import compute_errors

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
            depth = depth.to(device)
            errors = np.zeros((2, 9, len(test_files)), np.float32)
            mean_errors = errors.mean(2)

            gt = tifffile.imread('/home/zyd/respository/sfmlearner_results/endo_testset/left_depth_map_d4k1_000000.tiff')
            gt = gt[:, :, 2]

            abs_diff, abs_rel, sq_rel, a1, a2, a3 = 0,0,0,0,0,0
            if 1:
                crop_mask = gt[0] != gt[0]
                y1,y2 = int(0.40810811 * 1024), int(0.99189189 * 1024)
                x1,x2 = int(0.03594771 * 1280), int(0.96405229 * 1280)
                crop_mask[y1:y2,x1:x2] = 1

            for current_gt, current_pred in zip(gt, pred):
                valid = (current_gt > 0) & (current_gt < 80)
            if 1:
                valid = valid & crop_mask

            valid_gt = current_gt[valid]
            valid_pred = current_pred[valid].clamp(1e-3, 80)

            valid_pred = valid_pred * torch.median(valid_gt)/torch.median(valid_pred)

            thresh = torch.max((valid_gt / valid_pred), (valid_pred / valid_gt))
            a1 += (thresh < 1.25).float().mean()
            a2 += (thresh < 1.25 ** 2).float().mean()
            a3 += (thresh < 1.25 ** 3).float().mean()

            abs_diff += torch.mean(torch.abs(valid_gt - valid_pred))
            abs_rel += torch.mean(torch.abs(valid_gt - valid_pred) / valid_gt)

            sq_rel += torch.mean(((valid_gt - valid_pred)**2) / valid_gt)
            
            error_names = ['abs_diff', 'abs_rel','sq_rel','rms','log_rms', 'abs_log', 'a1','a2','a3']
            

            
            print("Results with scale factor determined by GT/prediction ratio (like the original paper) : ")
            print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format(*error_names))
            print("{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}".format(*mean_errors[1]))



             

if __name__ == '__main__':
    main()

import torchvision
import yaml
import argparse
from dataset import *
from torch.utils.data import DataLoader
from utils.dir_utils import mkdir, get_last_path
from utils.model_utils import load_checkpoint
from model.dkt_adapted import URSCT
from tqdm import tqdm
from utils.image_utils import torchPSNR, torchSSIM
import torchvision.transforms.functional as TF
import os
from dataset.data_loader_detail import *
def get_infer_data(dir, img_options):
    assert os.path.exists(dir)
    return DataLoaderInf(dir, img_options)
def get_dataloader(opt_test, mode):
    if mode == 'eval':
        loader = DataLoader(dataset=get_test_data(opt_test['TEST_DIR'], {'patch_size': opt_test['TEST_PS']}),
                   batch_size=1, shuffle=False, num_workers=opt_test['NUM_WORKS'])
    elif mode == 'infer':
        loader = DataLoader(dataset=get_infer_data('/content/Underwater-image-enhancement_using_DKT/dataset/demo_data_Enh/test_data', {'patch_size': opt_test['TEST_PS']}),
                   batch_size=1, shuffle=False, num_workers=opt_test['NUM_WORKS'])
    return loader

def main(test_loader, opt_test, mode):
    if mode == 'eval':
        PSNRs, SSIMs = [], []
        for i, data in enumerate(tqdm(test_loader)):
            input = data[0].to(device)
            target_SR = data[1].to(device)
            with torch.no_grad():
                restored_SR = model(input)
            PSNRs.append(torchPSNR(restored_SR,target_SR))
            SSIMs.append(torchSSIM(restored_SR, target_SR))
            torchvision.utils.save_image(torch.cat( (TF.resize(input[0],opt_test['TEST_PS']),
                                                     restored_SR[0],target_SR[0]), -1),
                                         os.path.join(result_dir, str(i) + '.png'))
        print(
            "[PSNR] mean: {:.4f} std: {:.4f}".format(torch.stack(PSNRs).mean().item(), torch.stack(PSNRs).std().item()))
        print(
            "[SSIM] mean: {:.4f} std: {:.4f}".format(torch.stack(SSIMs).mean().item(), torch.stack(SSIMs).std().item()))
    elif mode == 'infer':
        for i, data in enumerate(tqdm(test_loader)):
            input = data.to(device)
            with torch.no_grad():
                restored_SR = model(input)
            torchvision.utils.save_image(torch.cat((TF.resize(input[0], opt_test['TEST_PS']),
                                                    restored_SR[0]), -1),
                                         os.path.join(result_dir, str(i) + '.png'))


#if __name__ == '__main__':
   # parser = argparse.ArgumentParser()
    #parser.add_argument('--mode', type=str, default='infer', choices=['infer', 'eval'], help='random seed')
    #mode = parser.parse_args().mode
    #import sys
    #if '-f' in sys.argv:
     # sys.argv.remove('-f')
      #sys.argv.remove('/root/.local/share/jupyter/runtime/kernel-b8edc924-2d03-48d8-8b18-3cd9afda376d.json')  
      #mode = parser.parse_args().mode
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='infer', choices=['infer', 'eval'], help='random seed')
    import sys
    if '-f' in sys.argv:
        sys.argv.remove('-f')
    if '/root/.local/share/jupyter/runtime/kernel-d5f60b45-e76f-47f5-bc39-d582fb46404a.json' in sys.argv:
        sys.argv.remove('/root/.local/share/jupyter/runtime/kernel-d5f60b45-e76f-47f5-bc39-d582fb46404a.json')
    mode = parser.parse_args().mode

    with open('/content/Underwater-image-enhancement_using_DKT/configs/Enh_opt.yaml', 'r') as config:
        opt = yaml.safe_load(config)
        opt_test = opt['TEST']
    device = opt_test['DEVICE']
    model_detail_opt = opt['MODEL_DETAIL']
    result_dir = os.path.join('/content/Underwater-image-enhancement_using_DKT/exps', 'test_results')
    mkdir(result_dir)

   

    model = URSCT(model_detail_opt).to(device)
    #path_chk_rest = get_last_path(os.path.join(opt_test['SAVE_DIR'], opt['TRAINING']['MODEL_NAME'], 'models'), '_bestSSIM.pth')
    path_chk_rest= get_last_path('/content/Underwater-image-enhancement_using_DKT/exps/dkt_mod/models','_bestSSIM.pth')
    load_checkpoint(model, path_chk_rest)
    model.eval()

   # model_dir = os.path.join(opt_test['SAVE_DIR'], opt['TRAINING']['MODEL_NAME'], 'models')
    model_dir = opt_test['SAVE_DIR'] + '/' + opt['TRAINING']['MODEL_NAME'] + '/models'
    files = os.listdir('/content/Underwater-image-enhancement_using_DKT/exps/dkt_mod/models')
    filtered_files = [f for f in files if f.endswith('_bestSSIM.pth')]



    test_loader = get_dataloader(opt_test, mode)
    main(test_loader, opt_test, mode)

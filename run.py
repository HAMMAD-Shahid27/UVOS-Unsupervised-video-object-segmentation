from dataset import *
import evaluation
from vise import VISE
from trainer import Trainer
from optparse import OptionParser
import warnings
warnings.simplefilter("ignore")


parser = OptionParser()
parser.add_option('--train', action='store_true', default=None)
parser.add_option('--test', action='store_true', default=None)
options = parser.parse_args()[0]


def train_duts_davis(model, ver):
    duts_set = TrainDUTS('/home/uvos/Documents/TMO-main/TMO-main/DB/DUTS', clip_n=384)
    davis_set = TrainDAVIS('/home/uvos/Documents/TMO-main/TMO-main/DB/DAVIS', '2016', 'train', clip_n=180)
    train_set = torch.utils.data.ConcatDataset([duts_set, davis_set])
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=6, shuffle=True, num_workers=4, pin_memory=True)
    val_set = TestDAVIS('/home/uvos/Documents/TMO-main/TMO-main/DB/DAVIS', '2016', 'val')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    trainer = Trainer(model, ver, optimizer, train_loader, val_set, save_name='Conrastive_LOSS_mitb2_Adaptive(weights)', val_step=100)
    trainer.train(4000)


def test_davis(model):
    evaluator = evaluation.Evaluator(TestDAVIS('/home/uvos/Documents/TMO-main/TMO-main/DB/DAVIS', '2016', 'val'))
    evaluator.evaluate(model, os.path.join('outputs', 'DAVIS16_val'))


def test_fbms(model):
    test_set = TestFBMS('/home/uvos/Documents/TMO-main/TMO-main/DB/FBMS/TestSet')
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, num_workers=4)
    model.cuda()
    ious = []

    # inference
    for vos_data in test_loader:
        imgs = vos_data['imgs'].cuda()
        flows = vos_data['flows'].cuda()
        masks = vos_data['masks'].cuda()
        video_name = vos_data['video_name'][0]
        files = vos_data['files']
        os.makedirs('outputs/FBMS_test/{}'.format(video_name), exist_ok=True)
        vos_out = model(imgs, flows)

        # get iou of each sequence
        iou = 0
        count = 0
        for i in range(masks.size(1)):
            tv.utils.save_image(vos_out['masks'][0, i].float(), 'outputs/FBMS_test/{}/{}'.format(video_name, files[i][0].split('/')[-1]))
            if torch.sum(masks[0, i]) == 0:
                continue
            iou = iou + torch.sum(masks[0, i] * vos_out['masks'][0, i]) / torch.sum((masks[0, i] + vos_out['masks'][0, i]).clamp(0, 1))
            count = count + 1
        print('{} iou: {:.5f}'.format(video_name, iou / count))
        ious.append(iou / count)

    # calculate overall iou
    print('-------(FBMS)---\' IOU: {:.5f}\n'.format(sum(ious) / len(ious)))


def test_ytobj(model):
    test_set = TestYTOBJ('/home/uvos/Documents/TMO-main/TMO-main/DB/YTOBJ')
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, num_workers=4)
    model.cuda()
    
    ious = {cls: [] for cls in ['aeroplane', 'bird', 'boat', 'car', 'cat', 
                                'cow', 'dog', 'horse', 'motorbike', 'train']}
    
    total_iou = 0
    total_count = 0

    # Inference
    for vos_data in test_loader:
        imgs = vos_data['imgs'].cuda()
        flows = vos_data['flows'].cuda()
        masks = vos_data['masks'].cuda()
        class_name = vos_data['class_name'][0]
        video_name = vos_data['video_name'][0]
        files = vos_data['files']
        
        os.makedirs(f'outputs/YTOBJ/{class_name}/{video_name}', exist_ok=True)
        vos_out = model(imgs, flows)

        # Get IoU of each sequence
        iou = 0
        count = 0
        for i in range(masks.size(1)):
            tv.utils.save_image(vos_out['masks'][0, i].float(), 
                                f'outputs/YTOBJ/{class_name}/{video_name}/{files[i][0].split("/")[-1]}')
            if torch.sum(masks[0, i]) == 0:
                continue
            
            iou += torch.sum(masks[0, i] * vos_out['masks'][0, i]) / \
                   torch.sum((masks[0, i] + vos_out['masks'][0, i]).clamp(0, 1))
            count += 1
        
        if count == 0:
            continue
        
        ious[class_name].append(iou / count)
        total_iou += iou / count
        total_count += 1

    
    print("\n===== Class-wise IoU Scores (%) =====")
    for class_name, scores in ious.items():
        if scores: 
            class_iou = sum(scores) / len(scores) * 100
            print(f"{class_name}: {class_iou:.1f}%")

    
    if total_count > 0:
        total_iou_percentage = (total_iou / total_count) * 100
        print("\n===== Overall IoU Score (%) =====")
        print(f"Total YTOBJ IoU: {total_iou_percentage:.1f}%\n")


def test_lvid(model):
    test_set = TestLVID('/home/uvos/Documents/TMO-main/TMO-main/DB/LVID')
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, num_workers=4)
    model.cuda()
    ious = []

   
    for vos_data in test_loader:
        imgs = vos_data['imgs'].cuda()
        flows = vos_data['flows'].cuda()
        masks = vos_data['masks'].cuda()
        video_name = vos_data['video_name'][0]
        files = vos_data['files']
        os.makedirs('outputs/LVID/{}'.format(video_name), exist_ok=True)
        vos_out = model(imgs, flows)

        
        iou = 0
        count = 0
        for i in range(masks.size(1)):
            tv.utils.save_image(vos_out['masks'][0, i].float(), 'outputs/LVID/{}/{}'.format(video_name, files[i][0].split('/')[-1]))
            if torch.sum(masks[0, i]) == 0:
                continue
            iou = iou + torch.sum(masks[0, i] * vos_out['masks'][0, i]) / torch.sum((masks[0, i] + vos_out['masks'][0, i]).clamp(0, 1))
            count = count + 1
        print('{} iou: {:.5f}'.format(video_name, iou / count))
        ious.append(iou / count)

    
    print('----------------------total seqs:LVID\' iou: {:.5f}\n'.format(sum(ious) / len(ious)))


if __name__ == '__main__':

    
    torch.cuda.set_device(0)

    
    ver = 'mitb2'
    model = TMO(ver).eval()

    
    if options.train:
        model = torch.nn.DataParallel(model)
        train_duts_davis(model, ver)

   
    if options.test:
        model.load_state_dict(torch.load(''.format(ver), map_location='cpu'))
        with torch.no_grad():
            test_davis(model)
            test_fbms(model)
            test_ytobj(model)
            test_lvid(model)

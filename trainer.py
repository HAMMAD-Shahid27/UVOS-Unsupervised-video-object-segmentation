from evaluation import metrics
from utils import AverageMeter, get_iou
import copy
import numpy
import torch
from tqdm import tqdm

class Trainer(object):
    def __init__(self, model, ver, optimizer, train_loader, val_set, save_name, val_step, lambda_contrast=0.1, output_dir=None):
        self.model = model.cuda()
        self.ver = ver
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_set = val_set
        self.save_name = save_name
        self.val_step = val_step
        self.lambda_contrast = lambda_contrast 
        self.epoch = 1
        self.best_score = 0
        self.score = 0
        self.stats = {'loss': AverageMeter(), 'contrastive_loss': AverageMeter(), 'iou': AverageMeter()}
        self.output_dir = output_dir  
        self.best_jm = 0  
        self.best_fm = 0

    
    def train(self, max_epochs):
        for epoch in range(self.epoch, max_epochs + 1):
            self.epoch = epoch
            self.train_epoch()
            

            if self.epoch % self.val_step == 0:
                with torch.no_grad():
                    self.score = self.cycle_dataset(mode='val')


            
            if self.score > self.best_score:
                print("-----------------------------------------------------------")
                print(f"New best score {self.score:.5f} at epoch {self.epoch}. Saving best model")
                print("-----------------------------------------------------------")
                self.best_score = self.score
                self.save_checkpoint(alt_name='best')

        print('Training finished!')
        print(f"Best Score: {self.best_score:.5f}")


    def train_epoch(self):
        if self.ver != 'rn101':
            self.model.train()
        self.cycle_dataset(mode='train')
        for stat_value in self.stats.values():
            stat_value.new_epoch()

    def cycle_dataset(self, mode):
        if mode == 'train':
            for vos_data in self.train_loader:
                imgs = vos_data['imgs'].cuda()
                flows = vos_data['flows'].cuda()
                indices = vos_data['indices'].cuda()
                masks = vos_data['masks'].cuda()
                B, L, _, H, W = imgs.size()

                flows = indices * flows + (1 - indices) * imgs

                
                vos_out = self.model(imgs, flows)
                contrastive_loss = vos_out['contrastive_loss'] 

                segmentation_loss = torch.nn.CrossEntropyLoss()(
                    vos_out['scores'].view(B * L, 2, H, W), masks.reshape(B * L, H, W)
                )

                total_loss = segmentation_loss + self.lambda_contrast * contrastive_loss

                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

                self.stats['loss'].update(segmentation_loss.detach().cpu().item(), B)
                self.stats['contrastive_loss'].update(contrastive_loss.detach().cpu().item(), B)
                iou = torch.mean(get_iou(vos_out['scores'].view(B * L, 2, H, W), masks.reshape(B * L, H, W))[:, 1:])
                self.stats['iou'].update(iou.detach().cpu().item(), B)

            print(f'[Epoch {self.epoch:04d}] Cross-Entropy Loss: {self.stats["loss"].avg:.5f}, Contrastive Loss: {self.stats["contrastive_loss"].avg:.5f}, IoU: {self.stats["iou"].avg:.3f}')

        if mode == 'val':
            with torch.no_grad():
                metrics_res = {'J': [], 'F': []}

                for video_name, video_parts in self.val_set.get_videos():
                    for vos_data in tqdm(video_parts, desc=f"Processing video {video_name}", leave=False,dynamic_ncols=True):
                        imgs = vos_data['imgs'].cuda()
                        flows = vos_data['flows'].cuda()
                        masks = vos_data['masks'].cuda()

                        vos_out = self.model(imgs, flows)
                        res_masks = vos_out['masks'][:, 1:-1].squeeze(2)
                        gt_masks = masks[:, 1:-1].squeeze(2)
                        B, L, H, W = res_masks.shape
                        object_ids = numpy.unique(gt_masks.cpu()).tolist()
                        object_ids.remove(0)

                        all_res_masks = numpy.zeros((len(object_ids), L, H, W))
                        all_gt_masks = numpy.zeros((len(object_ids), L, H, W))
                        for k in object_ids:
                            res_masks_k = copy.deepcopy(res_masks).cpu().numpy()
                            res_masks_k[res_masks_k != k] = 0
                            res_masks_k[res_masks_k != 0] = 1
                            all_res_masks[k - 1] = res_masks_k[0]
                            gt_masks_k = copy.deepcopy(gt_masks).cpu().numpy()
                            gt_masks_k[gt_masks_k != k] = 0
                            gt_masks_k[gt_masks_k != 0] = 1
                            all_gt_masks[k - 1] = gt_masks_k[0]

                        j_metrics_res = numpy.zeros(all_res_masks.shape[:2])
                        f_metrics_res = numpy.zeros(all_res_masks.shape[:2])
                        for i in range(all_res_masks.shape[0]):
                            j_metrics_res[i] = metrics.db_eval_iou(all_gt_masks[i], all_res_masks[i])
                            f_metrics_res[i] = metrics.db_eval_boundary(all_gt_masks[i], all_res_masks[i])
                            [JM, _, _] = metrics.db_statistics(j_metrics_res[i])
                            metrics_res['J'].append(JM)
                            [FM, _, _] = metrics.db_statistics(f_metrics_res[i])
                            metrics_res['F'].append(FM)

                J, F = metrics_res['J'], metrics_res['F']
                print("------------------------------")
                print(f"Validation - Mean J: {numpy.mean(J):.3f}, Mean F: {numpy.mean(F):.3f}, Combined J&F: {(numpy.mean(J) + numpy.mean(F)) / 2:.3f}")

                if numpy.mean(J) > self.best_jm:
                    self.best_jm = numpy.mean(J)
                if numpy.mean(F) > self.best_fm:
                    self.best_fm = numpy.mean(F)

                return (numpy.mean(J) + numpy.mean(F)) / 2

    
    def save_checkpoint(self, alt_name=None):
        """Save the best model checkpoint."""
        if alt_name is not None:
            file_path = 'weights/{}_{}.pth'.format(self.save_name, alt_name)
        else:
            file_path = 'weights/{}_{:04d}.pth'.format(self.save_name, self.epoch)
        
        torch.save(self.model.module.state_dict(), file_path)
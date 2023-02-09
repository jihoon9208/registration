import torch

def transformation_loss( gt_R, gt_t, pred_R, pred_t):
    '''
    Input:
        pred_R: [B,3,3]
        pred_t: [B,3]
        gt_R: [B,3,3]
        gt_t: [B,3]
        alpha: weight
    '''
    Identity = torch.eye(3,3).cuda()

    resi_R = torch.norm((torch.matmul(pred_R.transpose(1,0).contiguous(),gt_R) - Identity), dim=0, keepdim=False)
    resi_t = torch.norm((pred_t - gt_t),  keepdim=False)
    loss_R = torch.mean(resi_R)
    loss_t = torch.mean(resi_t)
    
    return loss_R, loss_t


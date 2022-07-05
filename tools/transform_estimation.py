import torch
import MinkowskiEngine as ME


def rot_x(x):
    out = torch.zeros((3, 3))
    c = torch.cos(x)
    s = torch.sin(x)
    out[0, 0] = 1
    out[1, 1] = c
    out[1, 2] = -s
    out[2, 1] = s
    out[2, 2] = c
    return out


def rot_y(x):
    out = torch.zeros((3, 3))
    c = torch.cos(x)
    s = torch.sin(x)
    out[0, 0] = c
    out[0, 2] = s
    out[1, 1] = 1
    out[2, 0] = -s
    out[2, 2] = c
    return out


def rot_z(x):
    out = torch.zeros((3, 3))
    c = torch.cos(x)
    s = torch.sin(x)
    out[0, 0] = c
    out[0, 1] = -s
    out[1, 0] = s
    out[1, 1] = c
    out[2, 2] = 1
    return out


def get_trans(x):
    trans = torch.eye(4)
    trans[:3, :3] = rot_z(x[2]).mm(rot_y(x[1])).mm(rot_x(x[0]))
    trans[:3, 3] = x[3:, 0]
    return trans


def update_pcd(pts, trans):
    R = trans[:3, :3]
    T = trans[:3, 3]
    # pts = R.mm(pts.t()).t() + T.unsqueeze(1).t().expand_as(pts)
    pts = torch.t(R @ torch.t(pts)) + T
    return pts


def build_linear_system(pts0, pts1, weight):
    npts0 = pts0.shape[0]
    A0 = torch.zeros((npts0, 6))
    A1 = torch.zeros((npts0, 6))
    A2 = torch.zeros((npts0, 6))
    A0[:, 1] = pts0[:, 2]
    A0[:, 2] = -pts0[:, 1]
    A0[:, 3] = 1
    A1[:, 0] = -pts0[:, 2]
    A1[:, 2] = pts0[:, 0]
    A1[:, 4] = 1
    A2[:, 0] = pts0[:, 1]
    A2[:, 1] = -pts0[:, 0]
    A2[:, 5] = 1
    ww1 = weight.repeat(3, 6)
    ww2 = weight.repeat(3, 1)
    A = ww1 * torch.cat((A0, A1, A2), 0)
    b = ww2 * torch.cat(
        (pts1[:, 0] - pts0[:, 0], pts1[:, 1] - pts0[:, 1], pts1[:, 2] - pts0[:, 2]),
        0,
    ).unsqueeze(1)
    return A, b


def solve_linear_system(A, b):
    temp = torch.inverse(A.t().mm(A))
    return temp.mm(A.t()).mm(b)


def compute_weights(pts0, pts1, par):
    return par / (torch.norm(pts0 - pts1, dim=1).unsqueeze(1) + par)


def est_quad_linear_robust(pts0, pts1, weight=None):
    # TODO: 2. residual scheduling
    pts0_curr = pts0
    trans = torch.eye(4)

    par = 1.0  # todo: need to decide
    if weight is None:
        weight = torch.ones(pts0.size()[0], 1)

    for i in range(20):
        if i > 0 and i % 5 == 0:
            par /= 2.0

        # compute weights
        A, b = build_linear_system(pts0_curr, pts1, weight)
        x = solve_linear_system(A, b)

        # TODO: early termination
        # residual = np.linalg.norm(A@x - b)
        # print(residual)

        # x = torch.empty(6, 1).uniform_(0, 1)
        trans_curr = get_trans(x)
        pts0_curr = update_pcd(pts0_curr, trans_curr)
        weight = compute_weights(pts0_curr, pts1, par)
        trans = trans_curr.mm(trans)

    return trans


def pose_estimation(model,
                    device,
                    xyz0,
                    xyz1,
                    coord0,
                    coord1,
                    feats0,
                    feats1,
                    return_corr=False):
    sinput0 = ME.SparseTensor(feats0.to(device), coordinates=coord0.to(device))
    F0 = model(sinput0).F

    sinput1 = ME.SparseTensor(feats1.to(device), coordinates=coord1.to(device))
    F1 = model(sinput1).F

    corr = F0.mm(F1.t())
    weight, inds = corr.max(dim=1)
    weight = weight.unsqueeze(1).cpu()
    xyz1_corr = xyz1[inds, :]

    trans = est_quad_linear_robust(xyz0, xyz1_corr, weight)  # let's do this later

    if return_corr:
        return trans, weight, corr
    else:
        return trans, weight

import MinkowskiEngine as ME
import torch

eps = 1e-8


def is_rotation(R):
    assert R.shape == (
        3,
        3,
    ), f"rotation matrix should be in shape (3, 3) but got {R.shape} input."
    rrt = R @ R.t()
    I = torch.eye(3)
    err = torch.norm(I - rrt)
    return err < eps


def skew_symmetric(vectors):
    if vectors.dim() == 1:
        vectors = vectors.unsqueeze(0)

    r00 = torch.zeros_like(vectors[:, 0])
    r01 = -vectors[:, 2]
    r02 = vectors[:, 1]
    r10 = vectors[:, 2]
    r11 = torch.zeros_like(r00)
    r12 = -vectors[:, 0]
    r20 = -vectors[:, 1]
    r21 = vectors[:, 0]
    r22 = torch.zeros_like(r00)

    R = torch.stack([r00, r01, r02, r10, r11, r12, r20, r21, r22], dim=-1).reshape(
        -1, 3, 3
    )
    return R


def axis_angle_to_rotation(axis_angles):
    if axis_angles.dim() == 1:
        axis_angles = axis_angles.unsqueeze(0)

    angles = torch.norm(axis_angles, p=2, dim=-1, keepdim=True)
    axis = axis_angles / angles

    K = skew_symmetric(axis)
    K_square = torch.bmm(K, K)
    I = torch.eye(3).to(axis_angles.device).repeat(K.shape[0], 1, 1)

    R = (
        I
        + torch.sin(angles).unsqueeze(-1) * K
        + (1 - torch.cos(angles).unsqueeze(-1)) * K_square
    )

    return R.squeeze(0)


def rotation_to_axis_angle(R):
    if R.dim() == 2:
        R = R.unsqueeze(0)

    theta = torch.acos(((R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]) - 1) / 2 + eps)
    sin_theta = torch.sin(theta)

    singular = torch.zeros(3, dtype=torch.float32).to(theta.device)

    multi = 1 / (2 * sin_theta + eps)
    rx = multi * (R[:, 2, 1] - R[:, 1, 2]) * theta
    ry = multi * (R[:, 0, 2] - R[:, 2, 0]) * theta
    rz = multi * (R[:, 1, 0] - R[:, 0, 1]) * theta

    axis_angles = torch.stack((rx, ry, rz), dim=-1)
    singular_indices = torch.logical_or(sin_theta == 0, sin_theta.isnan())
    axis_angles[singular_indices] = singular

    return axis_angles.squeeze(0)


def gaussianNd(kernel_size=5, dimension=3):
    dim = [kernel_size] * dimension
    siz = torch.LongTensor(dim)
    sig_sq = (siz.float() / 2 / 2.354).pow(2)
    siz2 = torch.div((siz - 1) , 2, rounding_mode="floor")

    axis = torch.meshgrid(
        [torch.arange(-siz2[i], siz2[i] + 1) for i in range(siz.shape[0])]
    )
    gaussian = torch.exp(
        -torch.stack(
            [axis[i].float().pow(2) / 2 / sig_sq[i] for i in range(sig_sq.shape[0])],
            dim=0,
        ).sum(dim=0)
    )
    gaussian = gaussian / gaussian.sum()
    return gaussian


def sparse_gaussian(data, kernel_size=5, dimension=3):
    # prepare input sparse tensor
    if isinstance(data, ME.SparseTensor):
        sinput = data
    else:
        raise TypeError()

    # build gaussian kernel weight
    hsfilter = gaussianNd(kernel_size, dimension).to(data.device)

    # prepare conv layer
    conv = ME.MinkowskiConvolution(
        in_channels=1, out_channels=1, kernel_size=kernel_size, dimension=dimension
    )
    with torch.no_grad():
        conv.kernel.data = hsfilter.reshape(-1, 1, 1)

    # forward
    out = conv(sinput)
    return out


def sparse_bilateral(data, kernel_size=5, dimension=3):
    # prepare input sparse tensor
    if isinstance(data, ME.SparseTensor):
        sinput = data
    else:
        raise TypeError()

    # build gaussian kernel weight
    hsfilter = gaussianNd(kernel_size, dimension).to(data.device)

    # prepare conv layer
    conv = ME.MinkowskiConvolution(
        in_channels=1, out_channels=1, kernel_size=kernel_size, dimension=dimension
    )
    with torch.no_grad():
        conv.kernel.data = hsfilter.reshape(-1, 1, 1)

    # forward
    out = conv(sinput)
    return out
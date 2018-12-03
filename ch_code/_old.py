def get_readout(x_hat):
    """
    """
    readout = x_hat[...,:3]
    gt_one  = (F.sign(readout - 1) + 1) // 2
    ls_zero = -(F.sign(readout) - 1) // 2
    rest = 1 - gt_one - ls_zero
    readout_xhat = rest*readout + gt_one*(readout-1) + ls_zero*(1-readout)
    return readout_xhat


def get_bounded_MSE(x_hat, x_true, boundary):
    x_hat_loc  = x_hat[...,:3]
    x_true_loc = x_true[...,:3]
    bidx = get_bounded(x_true_loc, boundary)
    bhat  = F.get_item(x_hat_loc, bidx)
    btrue = F.get_item(x_true_loc, bidx)
    return F.mean(F.sum(F.squared_difference(bhat, btrue), axis=-1))

def get_bounded_squared_error(x_hat, x_true, boundary):
    x_hat_loc  = x_hat[...,:3]
    x_true_loc = x_true[...,:3]
    bidx = get_bounded(x_true_loc, boundary)
    bhat  = F.get_item(x_hat_loc, bidx)
    btrue = F.get_item(x_true_loc, bidx)
    return F.squared_difference(bhat, btrue)

def get_bounded_MSE_vel(x_hat, x_true, boundary):
    #x_hat_loc  = x_hat[...,:3]
    x_true_loc = x_true[...,:3]
    bidx = get_bounded(x_true_loc, boundary)
    bhat  = F.get_item(x_hat, bidx)
    btrue = F.get_item(x_true, bidx)
    mse_loc = F.mean(F.sum(F.squared_difference(bhat[...,:3], btrue[...,:3]), axis=-1))
    mse_vel = F.mean(F.sum(F.squared_difference(bhat[...,3:], btrue[...,3:]), axis=-1))
    return mse_loc*mse_vel, mse_loc, mse_vel


def get_combined_MSE(x_input, x_hat, x_true, boundary): # EXPERIMENTAL
    x_input_loc  = x_input[...,:3]
    x_hat_loc = x_hat[...,:3]
    x_true_loc = x_true[...,:3]
    bidx = get_bounded(x_true_loc, boundary)
    binput = F.get_item(x_input_loc, bidx)
    bhat   = F.get_item(x_hat_loc, bidx)
    btrue  = F.get_item(x_true_loc, bidx)

    dist_in_true  = F.sum(F.squared_difference(binput, btrue), axis=-1)
    dist_in_hat   = F.sum(F.squared_difference(binput,  bhat), axis=-1)
    dist_hat_true = F.sum(F.squared_difference(bhat,   btrue), axis=-1)
    input_dist = F.squared_difference(dist_in_true, dist_in_hat)
    combined = F.mean(input_dist * dist_hat_true)
    normal = F.mean(dist_hat_true.data).data

    return combined, normal



def mean_squared_error(x_hat, x_true, boundary=(0.095, 1-0.095)):
    if boundary is None:
        return get_min_readout_MSE(x_hat, x_true)
    else:
        return get_bounded_MSE(x_hat, x_true, boundary)


def get_min_readout_MSE(x_hat, x_true):
    '''x_hat needs to be bounded'''
    readout = get_readout(x_hat)
    x_true_loc = x_true[...,:3]
    dist = F.minimum(F.square(readout - x_true_loc), F.square(readout - (1 + x_true_loc)))
    dist = F.minimum(dist, F.square((1 + readout) - x_true_loc))
    mse = F.mean(F.sum(dist, axis=-1))
    return mse


import numpy as np
from scipy.signal import convolve
from scipy.special import factorial
from fastkde import fastKDE
def KM(timeseries: np.ndarray, powers: np.ndarray,bins: np.ndarray,
          tol: float=1e-7,correction: bool = False,
        conv_method: str='auto',norm: bool = False) -> np.ndarray:
    timeseries = np.asarray_chkfinite(timeseries, dtype=float)
    if len(timeseries.shape) == 1:
        timeseries = timeseries.reshape(-1, 1)

    assert len(timeseries.shape) == 2, "Timeseries must (n, dims) shape"
    assert timeseries.shape[0] > 0, "No data in timeseries"
    assert timeseries.shape[0] > timeseries.shape[1] , "Timeseries must (n, dims) shape,not(dims, n)"

    n, dims = timeseries.shape

    power_max=np.max(powers)

    Powers=[]
    for i in range(dims):
        power_=np.zeros((power_max+1,dims))
        power_[:,i]=np.arange(0,power_max+1,1)
        Powers.append(power_)  
    Powers=(np.array(Powers)).reshape(-1,dims)


    powers = np.asarray_chkfinite(powers, dtype=float)
    if len(powers.shape) == 1:
        powers = powers.reshape(-1, 1)

    if not (powers[0] == [0] * dims).all():
        powers = np.array([[0] * dims, *powers])
        trim_output = True
    else:
        trim_output = False

    assert (powers[0] == [0] * dims).all(), "First power must be zero"
    assert dims == powers.shape[1], "Powers not matching timeseries' dimension"
    assert dims == bins.shape[0], "Bins not matching timeseries' dimension"

    kmc, edges =  _km(timeseries, Powers,bins, tol, conv_method)
    edges=np.squeeze(np.array(edges))
    
    if correction == True:
        kmc = corrections(dims=dims,m = kmc, Power = Powers)

    matching_indices = find_matching_rows(Powers.tolist(), powers.tolist())
    kmc=kmc[matching_indices,:]
    if norm == True:
        taylors = np.prod(factorial(powers[1:]), axis=1)
        kmc[1:, :]= kmc[1:, :] / taylors[..., None]

    return (kmc, edges) if not trim_output else (kmc[1:], edges)


def _km(timeseries: np.ndarray, Powers: np.ndarray,bins: np.ndarray,
         tol: float, conv_method: str) -> np.ndarray:
 
    # Calculate derivative and the product of its powers
    grads = np.diff(timeseries, axis=0)
    weights = np.prod(np.power(grads.T, Powers[..., None]), axis=1)


    if timeseries[:-1,...].shape[1]==1:
        hist=[]
        for i in range(weights.shape[0]):
            hist1, edges= np.histogramdd(timeseries[:-1, ...], bins=bins,weights=weights[i])#
            hist.append(hist1)
        hist=np.array(hist)          
    elif timeseries[:-1,...].shape[1]==2:
        hist=[]
        for i in range(weights.shape[0]):
            hist1, edges= np.histogramdd(timeseries[:-1, ...], bins=bins,weights=weights[i])
            hist.append(hist1)
        hist=np.array(hist)
    elif timeseries[:-1,...].shape[1]==3:
        hist=[]
        for i in range(weights.shape[0]):
            hist1, edges= np.histogramdd(timeseries[:-1, ...], bins=bins,weights=weights[i])
            hist.append(hist1)
        hist=np.array(hist)
    elif timeseries.shape[1]>3:
            print('In practice, the situation of more than three-dimensional systems is not considered.')  
        
    if timeseries.shape[1]==1:

        kernel_, RX=fastKDE.pdf(np.squeeze(timeseries[:-1,0]),axisExpansionFactor=0)


    elif timeseries.shape[1]==2:

        kernel_, RX=fastKDE.pdf(np.squeeze(timeseries[:-1,0]),np.squeeze(timeseries[:-1,1]),axisExpansionFactor=0)


    elif timeseries.shape[1]==3:

        kernel_, RX=fastKDE.pdf(np.squeeze(timeseries[:-1,0]),np.squeeze(timeseries[:-1,1]),np.squeeze(timeseries[:-1,2]),axisExpansionFactor=0)

    elif timeseries.shape[1]>3:
        print('In practice, the situation of more than three-dimensional systems is not considered.')  

    # Convolve weighted histogram with kernel and trim it
    kmc = convolve(hist, kernel_[None, ...], mode='same', method=conv_method)
    # Normalise
    mask = np.abs(kmc[0]) < tol
    kmc[0:, mask] = 0.0
    kmc[1:, ~mask] /= kmc[0, ~mask]

    edgeee=[]
    for i in range(len(edges)):
        edgee=edges[i][:-1]#+ 0.5 * (edges[i][1] - edges[i][0])
        edgeee.append(edgee)
    return kmc,edgeee

def corrections(dims,m: np.ndarray, Power):
    F = np.zeros_like(m)
    ip=np.max(Power).astype(int)
    def correct(m,i:int):
        F[0+i] =m[0+i]
        F[1+i] =m[1+i]
        F[2+i] =( m[2+i] - (m[1+i]**2) )
        F[3+i] =( m[3+i] - 3*m[1+i]*m[2+i] + 2*(m[1+i]**3) )
        F[4+i] =( m[4+i] - 4*m[1+i]*m[3+i]- 3*(m[2+i]**2) \
                  + 12*(m[1+i]**2)*m[2+i] - 6*(m[1+i]**4) )
        F[5+i] =( m[5+i] - 5*m[1+i]*m[4+i]- 10*m[2+i]*m[3+i] + 30*m[1+i]*(m[2+i]**2) + 20*(m[1+i]**2)*m[3+i] - 60*(m[1+i]**3)*m[2+i]+ 24*(m[1+i]**5) )
        F[6+i] =( m[6+i] - 6*m[1+i]*m[5+i]- 10*(m[3+i]**2) - 15*m[2+i]*m[4+i] +30*(m[2+i]**3)+ 120*m[1+i]*m[2+i]*m[3+i]+30*(m[1+i]**2)*m[4+i]\
                 -270*(m[1+i]**2)*(m[2+i]**2)-120*(m[1+i]**3)*m[3+i]\
                     +360*(m[1+i]**4)*m[2+i]-120*(m[1+i]**6) )
        return F
    if dims==1:
        F1=correct(m,i=0)
    if dims==2:
        F1=correct(m,i=0)
        F1=correct(m,i=ip+1)
    if dims==3:
        F1=correct(m,i=0)
        F1=correct(m,i=ip+1)
        F1=correct(m,i=(ip+1)*2)
    return F1

def find_matching_rows(A, B):
    indices = []
    for i, row_B in enumerate(B):
        for j, row_A in enumerate(A):
            if row_B == row_A:
                indices.append(j)
                break
    return indices
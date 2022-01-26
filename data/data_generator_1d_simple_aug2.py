import torch
import collections
import numpy as np
import random
'''
The input to NP is NPRegressionDescription namedTuple. The fields are
query: A tuple having ((context_x, context_y), target_x)
target_y: A tensor containing gt for targets to be predicted
num_total_points: A vector containing scalar describing total (context + target)
number of data points
num_context_points: A vector containing scalar describint number of context points 
'''
NPRegressionDescription = collections.namedtuple(
    "NPRegressionDescription",
    ("query", "target_y", "num_total_points", "num_context_points","task_defn")
)

# GPCurvesReader returns newly sampled data in the above format at each iteration
class GPCurvesReader(object):
    '''
    Generate curves using a Gaussian Process (GP)

    Supports vector inputs (x) and vector outputs (y).

    Kernel: Mean-Squared Exponential kernel. x_value l2 coordinate distance scaled by
    some random factor consen randomly in a range
    Outputs: Independent Gaussian Processes
    '''

    def __init__(self,
                 batch_size,
                 max_num_context,
                 x_size=1,
                 y_size=1,
                 l1_scale=0.6,
                 sigma_scale=1.0,
                 testing=False,
                 random_kernel_parameters=False,
                 device="cpu"):
        '''
        Create a regression dataset of functions sampled from  a GP
        :param batch_size: int
        :param max_num_context: Maximum number of observations in the context set
        :param x_size: int, >=1, leangth of x values vector
        :param y_size: int, >=1, leangth of y values vector
        :param l1_scale: Float, scale for kernel distance fucntion
        :param sigma_scale: Float, scale for variance
        :param testing: Boolean, indicates whether we are testing. In testing, we have more targets
                        for visualization
        :random_kernel_parameters -> FOR Attentive Neural Processes (the kernel parameters (l1 and sigma) are
        randomized at each iteration)
        '''
        self._batch_size = batch_size
        self._max_num_context = max_num_context
        self._x_size = x_size
        self._y_size = y_size
        self._l1_scale = l1_scale
        self._sigma_scale = sigma_scale
        self._testing = testing
        self._random_kernel_parameters = random_kernel_parameters
        self._device=device

    def _gaussian_kernel(self,
                         xdata,
                         l1,
                         sigma_f,
                         sigma_noise=2e-2):
        '''
        Apply the gaussian kernel to generate curve data
        :param xdata: shape [batch_size, num_total_points, x_size] value of x_axis data
        :param l1: shape [batch_size, y_size, x_size], scale parameter of the GP
        :param sigma_f: Float tensor shape [batch_size, y_size], magnitude of std
        :param sigma_noise: Float, std of noise for stability
        :return:
        The kernel, a float tensor of shape [batch_size, y_size, num_total_pints, num_total_points]
        '''

        num_total_points = xdata.shape[1]

        #Expand and take difference
        xdata1 = xdata.unsqueeze(dim=1)
        xdata2 = xdata.unsqueeze(dim=2)

        diff = xdata1 - xdata2

        norm = (diff[:, None,:,:,:]/ l1[:,:,None,None,:]) ** 2
        norm = norm.sum(dim = -1)
        kernel = (sigma_f ** 2)[:,:,None,None] * torch.exp(-0.5 * norm)
        kernel += (sigma_noise**2) * torch.eye(num_total_points)
        return kernel

    def generate_curves(self, device, fixed_num_context=-1):
        '''
        Generate the curves
        x: float lies in range -2,2
        :return: A CNPRegressionDescription namedTuple
        '''
        if fixed_num_context > 0:
            num_context = fixed_num_context
        else:
            num_context = torch.randint(low=3, high=self._max_num_context + 1, size=(1,))

        # If testing, then more targets, evenly distributed at x values
        #For plotting
        if self._testing:
            num_target = 400
            num_total_points = num_target
            x_values = torch.linspace(start=-2,end = 2, steps=num_target)
            x_values = (
                x_values.unsqueeze(dim=0).repeat([self._batch_size,1]).unsqueeze(-1)
            )

        else:
            # num_context = 2
            num_target = torch.randint(3,self._max_num_context+1, size = (1,))
            num_total_points = num_target + num_context
            x_values = torch.rand([self._batch_size, num_total_points,self._x_size])*4-2

        #For NP and CNP, fixed kernel parameters
        l1 = torch.ones(self._batch_size,self._y_size,self._x_size) * self._l1_scale
        sigma_f = torch.ones(self._batch_size, self._y_size) * self._sigma_scale

        #FOR ANP we randomize the kernel parameters at each iteration
        if self._random_kernel_parameters:
            l1 = torch.rand([self._batch_size, self._y_size, self._x_size])*(self._l1_scale-0.1)+0.1
            sigma_f = torch.rand(self._batch_size, self._y_size)*(self._l1_scale-0.1)+0.1

        #Pass x values through the gaussian kernel
        #Return [batch size, y_size, num_total_points, num_total_points]
        kernel = self._gaussian_kernel(x_values,l1,sigma_f)

        cholesky = torch.cholesky(kernel.type(torch.DoubleTensor)).type(torch.FloatTensor)

        # [batch size, num_total_points, y_size]
        y_values = torch.matmul(cholesky, torch.randn([self._batch_size, self._y_size, num_total_points, 1]))

        y_values = y_values.squeeze(3).permute(0,2,1)
        # y_values[:,200:300] = -100*torch.rand(y_values[:,200:300].shape) #* 0.1

        task_property = torch.Tensor([self._l1_scale,self._sigma_scale])

        if self._testing:
            target_x = x_values
            target_y = y_values

            idx = torch.randperm(num_target)
            context_x = x_values[:, idx[:num_context], :]
            context_y = y_values[:,idx[:num_context],:]
        else:
            target_x = x_values[:,:num_target+num_context,:]
            target_y = y_values[:,:num_target+num_context,:]

            context_x = x_values[:,:num_context,:]
            context_y = y_values[:,:num_context,:]


        context_x = context_x.to(device)
        context_y = context_y.to(device)
        target_x = target_x.to(device)
        target_y = target_y.to(device)
        task_property = task_property.to(device)

        query = ((context_x, context_y), target_x)

        return NPRegressionDescription(
            query=query,
            target_y=target_y,
            num_total_points=target_x.shape[1],
            num_context_points=num_context,
            task_defn=task_property
        )





class SinusoidCurve(object):
    '''
    Generate curves using a Sinusoid Function y = A Sin (b x + c)

    Supports vector inputs (x) and vector outputs (y).

    '''

    def __init__(self,
                 batch_size,
                 max_num_context,
                 x_size=1,
                 y_size=1,
                 testing=False,
                 device="cpu"):
        '''
        Create a regression dataset of functions sampled from  a GP
        :param batch_size: int
        :param max_num_context: Maximum number of observations in the context set
        :param x_size: int, >=1, leangth of x values vector
        :param y_size: int, >=1, leangth of y values vector
        :param l1_scale: Float, scale for kernel distance fucntion
        :param sigma_scale: Float, scale for variance
        :param testing: Boolean, indicates whether we are testing. In testing, we have more targets
                        for visualization
        :random_kernel_parameters -> FOR Attentive Neural Processes (the kernel parameters (l1 and sigma) are
        randomized at each iteration)
        '''
        self._batch_size = batch_size
        self._max_num_context = max_num_context
        self._x_size = x_size
        self._y_size = y_size
        self._testing = testing
        self._device=device

    def sinusoid_generator(self, x, A, p, f):
        return A* torch.sin(f*x + p)

    def generate_curves(self, device, fixed_num_context=-1):
        '''
        Generate the curves
        x: float lies in range -5,5
        :return: A CNPRegressionDescription namedTuple
        '''
        if fixed_num_context>0:
            num_context = fixed_num_context
        else:
            num_context = torch.randint(low=3, high=self._max_num_context + 1, size=(1,))


        # If testing, then more targets, evenly distributed at x values
        #For plotting
        if self._testing:
            num_total_points = 400
            num_target = num_total_points
            x_values = torch.linspace(start=-5,end = 5, steps=num_target)
            x_values = (
                x_values.unsqueeze(dim=0).repeat([self._batch_size,1]).unsqueeze(-1)
            )

        else:
            num_target = torch.randint(3,self._max_num_context+1, size = (1,))
            num_total_points = num_target + num_context
            x_values = torch.rand([self._batch_size, num_total_points,self._x_size])*10-5

        A = torch.rand(self._batch_size, self._y_size, 1) * (5 - 0.1) + 0.1
        # A = torch.ones(self._batch_size, self._y_size, 1) * (5 - 0.1) + 0.1
        ph = torch.rand(self._batch_size, self._y_size, 1) * (np.pi)
        f = 1 #torch.rand(self._batch_size, self._y_size, 1) * (10.0 - 5.0) + 5.0

        y_values = self.sinusoid_generator(x_values, A, ph, f)
        # print(y_values.shape)
        # y_values += (torch.rand(y_values.shape) -0.5)#* 0.5
        # y_values += torch.normal(mean=torch.zeros(y_values.shape),std=torch.ones(y_values.shape))*2
        A_one = torch.reshape(A,(self._batch_size, 1))
        ph_one = torch.reshape(ph,(self._batch_size, 1))
        task_property = torch.cat((A_one,ph_one),1)


        if self._testing:
            target_x = x_values
            target_y = y_values

            idx = torch.randperm(num_target)
            context_x = x_values[:, idx[:num_context], :]
            context_y = y_values[:,idx[:num_context],:]
        else:
            target_x = x_values[:,:num_target+num_context,:]
            target_y = y_values[:,:num_target+num_context,:]

            context_x = x_values[:,:num_context,:]
            context_y = y_values[:,:num_context,:]


        context_x = context_x.to(device)
        context_y = context_y.to(device)
        target_x = target_x.to(device)
        target_y = target_y.to(device)
        task_property = task_property.to(device)

        query = ((context_x, context_y), target_x)

        return NPRegressionDescription(
            query=query,
            target_y=target_y,
            num_total_points=target_x.shape[1],
            num_context_points=num_context,
            task_defn=task_property
        )

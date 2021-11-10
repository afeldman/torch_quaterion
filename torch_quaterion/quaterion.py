''' quaternion with torch backend '''
import numbers
import torch
import numpy as np

from apu.ast.property import ClassProperty

class Quaterion:
    def __init__(self, size:int=0, *args, **kwargs) -> None:

        if size <= 0:
            size = 1

        self.cpu = True

        if torch.cuda.is_available():
            self.quaterion = torch.from_numpy(Quaterion.one).float().cuda().repeat(size,1)
            self.cpu = False
        else:
            self.quaterion = torch.from_numpy(Quaterion.one).float().cpu().repeat(size,1)

    def is_unit(self) -> bool:
        return bool(torch.all(self.norm.bool()))

    @property
    def normalize(self):
        if self.is_unit():
            self.quaterion = torch.nn.functional.normalize(self.quaterion)

        return self

    def complex(self):
        data = self.quaterion[:,0:2].cpu().numpy()
        result = np.empty(data.shape[:-1], dtype=complex)
        result.real, result.imag=data[...,0], data[...,1]
        return result

    # length
    def __len__(self):
        return int(self.quaterion.shape[0])

    # addition
    def __rsub__(self, p):
        return self - p

    def __isub__(self, p):
        return self - p

    def __sub__(self, p):
        return self.alpha_sub(p=p)

    def alpha_sub(self, p, alpha:int=1):
        assert isinstance(alpha, numbers.Number), "alpha is not a number"
        assert self.shape == p.shape, "can not create a random quaterion matrix. Please add a proper integer number"
        
        if isinstance(p, Quaterion):
            self.quaterion = torch.sub(self.quaterion, p.quaterion, alpha=alpha)
        elif isinstance(p, numbers.Number):
            self.quaterion = torch.sub(self.quaterion, p, alpha=alpha)
        else:
            raise TypeError("value is of wrong type")

        return self

    # addition
    def __radd__(self, p):
        return self + p

    def __iadd__(self, p):
        return self + p

    def __add__(self, p):
        return self.alpha_add(p=p)

    def alpha_add(self, p, alpha:int=1):
        assert isinstance(alpha, numbers.Number), "alpha is not a number"
        assert self.shape == p.shape, "can not create a random quaterion matrix. Please add a proper integer number"
        
        if isinstance(p, Quaterion):
            self.quaterion = torch.add(self.quaterion, p.quaterion, alpha=alpha)

        elif isinstance(p, numbers.Number):
            self.quaterion = torch.add(self.quaterion, p, alpha=alpha) 
        else:
            raise TypeError("value is of wrong type")

        return self

    # division
    def __div__(self, p):
        if isinstance(p, Quaterion):
            self.quaterion = torch.div(self.quaterion,p.quaterion)
        elif isinstance(p, numbers.Number):
            self.quaterion = torch.div(self.quaterion,p)
        else:
            raise TypeError("data type not implemented")

        return self

    def __rdiv__(self, p):
        return self / p

    def __idiv__(self, p):
        return self / p

    def __truediv__(self, p):
        if isinstance(p, Quaterion):
            self.quaterion = torch.true_divide(self.quaterion,p.quaterion)
        elif isinstance(p, numbers.Number):
            self.quaterion = torch.true_divide(self.quaterion,p)
        else:
            raise TypeError("data type not implemented")

        return self
        

    def __rtruediv__(self, p):
        return self /p

    def __itruediv__(self, p):
        return self /p

    def __floordiv__(self, p):
        if isinstance(p, Quaterion):
            self.quaterion = torch.floor_divide(self.quaterion,p.quaterion)
        elif isinstance(p, numbers.Number):
            self.quaterion = torch.floor_divide(self.quaterion,p)
        else:
            raise TypeError("data type not implemented")

        return self

    # multiplication 
    def __imul__(self, p):
        return self * p

    def __rmul__(self, p):
        return self * p

    def __mul__(self, p):

        if isinstance(p, Quaterion):

            assert p.shape == self.shape, "incoming vector has to be the same dimension"
            
            # Compute outer product
            terms = torch.bmm(p.quaterion.view(-1, 4, 1), self.quaterion.view(-1, 1, 4))

            w = terms[:, 0, 0] - terms[:, 1, 1] - terms[:, 2, 2] - terms[:, 3, 3]
            x = terms[:, 0, 1] + terms[:, 1, 0] - terms[:, 2, 3] + terms[:, 3, 2]
            y = terms[:, 0, 2] + terms[:, 1, 3] + terms[:, 2, 0] - terms[:, 3, 1]
            z = terms[:, 0, 3] - terms[:, 1, 2] + terms[:, 2, 1] + terms[:, 3, 0]

            self.quaterion = torch.stack((w, x, y, z), dim=1).view(self.shape)

        elif isinstance(p, numbers.Number):
            self.quaterion = torch.mul(self.quaterion, p) 
        else:
            raise TypeError("value is of wrong type")

        return self

    def __matmul__(self, p):
        return torch.matmul(self, p)

    def __abs__(self):
        return self.norm

    @property
    def sq_sum(self):
        return (self.quaterion * self.quaterion).sum(dim=1).float()

    @property
    def norm(self):
        return self.quaterion.norm(p=2, dim=1, keepdim=True).float().detach()

    @property
    def magnitude(self):
        return self.norm

    # properties
    @property
    def shape(self):
        return self.quaterion.shape

    @property
    def np(self):
        return self.quaterion.cpu().numpy()

    @property
    def elements(self):
        return self.quaterion

    @property
    def real(self):
        return self.scalar

    @property
    def imaginary(self):
        return self.vector

    @property
    def vector(self):
        return self.quaterion[:,1:]

    @property
    def scalar(self):
        return self.quaterion[:,0]

    @classmethod
    def random(self, counter:int=1):
        assert isinstance(counter, int), "can not create a random quaterion matrix. Please add a proper integer number"

        if counter <= 0:
            counter = 1

        quat = Quaterion()

        if quat.cpu:
            quat.quaterion = torch.rand(counter, 4).float().cpu()
        else:
            quat.quaterion = torch.rand(counter, 4).float().cuda()

        return quat

    @ClassProperty
    @classmethod
    def zero(cls):
        return np.array([0.,0.,0.,0.], dtype=np.float32)
    
    @ClassProperty
    @classmethod
    def one(cls):
        return np.array([1.,0.,0.,0.], dtype=np.float32)  
    
    @ClassProperty
    @classmethod
    def x_(cls):
        return np.array([0.,1.,0.,0.], dtype=np.float32)
  
    @ClassProperty
    @classmethod
    def y_(cls):
        return np.array([0.,0.,1.,0.], dtype=np.float32)

    @ClassProperty
    @classmethod
    def z_(cls):
        return np.array([0.,0.,0.,1.], dtype=np.float32)
  
    @property
    def w(self):
        return self.quaterion[:,0].numpy()

    @property
    def x(self):
        return self.quaterion[:,1].numpy()

    @property
    def y(self):
        return self.quaterion[:,2].numpy()

    @property
    def z(self):
        return self.quaterion[:,3].numpy()

    def __str__(self):
        return f"size is {self.__len__()}\n\n"

    def __repr__(self):
        return f"size is {self.__len__()}\n\n"

    def __eq__(self, p):
        return self.quaterion.equal(p.quaterion)

    def __eq__(self, p):
        return not self == p

if __name__=="__main__":
    quat = Quaterion(2)
    quat2= Quaterion.random(2133)
    quat2.normalize
    print(quat==quat)

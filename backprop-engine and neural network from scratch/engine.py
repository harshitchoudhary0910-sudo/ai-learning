
import math
class Value:
    '''for storing single scalar values and their gradients''' 
    def __init__(self,data,children=(),op=''):
        self.data=data
        self.grad=0.0
        self._backward = lambda: None
        self._children=children
        self._op=op



    def __add__(self,other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data+other.data,children=(self,other),op='+')
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out


    
    def __mul__(self,other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data*other.data,children=(self,other),op='*') 

        def backward():
            self.grad+=other.data*out.grad
            other.grad+=self.data*out.grad
        out._backward=backward 
        return out


    def __pow__(self,other):
        assert isinstance(other,(int,float)), "only supporting int/float powers for now"
        out = Value(self.data**other,children=(self,),op=f'**{other}')

        def backward():
            self.grad+=other*(self.data**(other-1))*out.grad

        out._backward=backward
        return out

    

    def tanh(self):
        pos_exp = math.exp(self.data)
        neg_exp = math.exp(-self.data)
        out = Value((pos_exp - neg_exp) / (pos_exp + neg_exp), children=(self,), op='tanh')

        def backward():
            self.grad+=(1.0-out.data**2)*out.grad

        out._backward=backward
        return out

    def relu(self):
        out = Value(0 if self.data < 0 else self.data, children=(self,), op='ReLU')

        def backward():
            self.grad += (out.data > 0) * out.grad

        out._backward = backward
        return out

    def sigmoid(self):
        out=Value(1.0/(1.0+math.exp(-self.data)),children=(self,),op='sigmoid')
        def backward():
            self.grad += out.data * (1 - out.data) * out.grad
        out._backward = backward
        return out

    def exp(self):
        out = Value(math.exp(self.data), children=(self,), op='exp')

        def backward():
            self.grad += out.data * out.grad

        out._backward = backward
        return out

    

    def backward(self):

        # Build topological order iteratively to avoid RecursionError
        topo = []
        visited = set()
        stack = [(self, False)]

        while stack:
            v, processed = stack.pop()

            if processed:
                topo.append(v)
                continue

            if v not in visited:
                visited.add(v)
                stack.append((v, True))

                for child in v._children:
                    if child not in visited:
                        stack.append((child, False))

        # Apply the chain rule in reverse topological order
        self.grad = 1.0

        for v in reversed(topo):
            v._backward()
    
    def __neg__(self): # -self
        return self * -1

    def __radd__(self, other): # other + self
        return self + other

    def __sub__(self, other):
       
       return self + (-other)

    def __rsub__(self, other):
        
        return other + (-self)
  
    def __rmul__(self, other): # other * self
        return self * other

    def __truediv__(self, other): # self / other
        return self * other**-1

    def __rtruediv__(self, other): # other / self
        return other * self**-1

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"

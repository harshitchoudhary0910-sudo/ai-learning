
import math
class Value:
    '''for storing single scalar values and their gradients''' 
    def __init__(self,data,children=(),op=''):
        self.data=data
        self.grad=0.0
        self._backward = lambda: None
        self._children=children
        self._op=op

    def __repr__(self):
        return f"Value(data={self.data})"

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

    

    def tanh(self):
        pos_exp = math.exp(self.data)
        neg_exp = math.exp(-self.data)
        out = Value((pos_exp - neg_exp) / (pos_exp + neg_exp), children=(self), op='tanh')

        def backward():
            self.grad+=(1.0-out**2)*out.grad

        out._backward=backward
        return out

    

    def backward(self):

        # topological order all of the children in the graph
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1
        for v in reversed(topo):
            v._backward()

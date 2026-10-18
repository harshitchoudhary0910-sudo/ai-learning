from engine import Value
import random

class Neuron:
    def __init__(self,nin):
        self.weights=[Value(random.uniform(-1,1)) for _ in range(nin)]
        self.bias=Value(0)

    def __call__(self,x):
        out=sum(x1*w for x1,w in zip(x,self.weights))+self.bias
        return out.tanh()

    def parameters(self):
        return self.weights+[self.bias]

    def __repr__(self):
        pass

class Layer:
    def __init__(self,nn,nin):
        self.neurons=[Neuron(nin) for _ in range(nn)]

    def __call__(self,x):
        out=[n(x) for n in self.neurons]
        return out

    def parameters(self):
        p=[]
        for n in self.neurons:
            p.extend(n.parameters())
        return p

class MLP:

    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i+1], sz[i]) for i in range(len(nouts))]
                       
                           

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        p=[]
        for layer in self.layers:
            p.extend(layer.parameters())

        return p



        
nn=MLP(2,[4,4,2])
#print(len(nn.parameters()))

inputs=[[1,0],[0,1],[1,1],[0,0]]
outputs=[1,1,0,0]

nn = MLP(2, [4, 4, 2])

inputs = [[1, 0], [0, 1], [1, 1], [0, 0]]
outputs = [1, 1, 0, 0]

def train(inputs, epochs=1000):
    for epoch in range(epochs):
        # 1. Forward pass and calculate total loss
        loss = 0
        for i in range(len(inputs)):
            out = nn(inputs[i])
            loss += (out[0] - outputs[i])**2
            loss += (out[1] - outputs[i])**2

        # 2. Reset gradients BEFORE backpropagation
        for p in nn.parameters():
            p.grad = 0

        # 3. Backpropagation
        loss.backward()

        # 4. Update parameters
        for p in nn.parameters():
            p.data += -0.1 * p.grad

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.data}")

train(inputs)

print("Final outputs after training:" )

for i in range(len(inputs)):
    out = nn(inputs[i])
    print(f"Input: {inputs[i]}, Output: {[o.data for o in out]}")

 





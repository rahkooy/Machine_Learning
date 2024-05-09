# Implementing backward function for gradient computation from scratch

def backward (incoming_gradients):
    self.Tensor.grad = incoming_gradients
    
    for inp in self.inputs:
        if inp.grad_fn is not None:
            new_incoming_gradients=incoming_gradient*local_grad(self.Tensor, inp)
            inp.grad_fn.backward(new_incoming_gradients)
        else:
            pass
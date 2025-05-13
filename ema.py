# The main purpose of this class is to help with parameter stability, especially since the data is really noisy.
class EMA:
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {} # Hashmap for EMA parameters
        self.backup = {} # Hashmap of original parameters
        self.register()
    
    def register(self):
        """
        Adds model parameters to 'self.shadow' 
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        """
        Calculates the moving average of the parameter and assigns it in 'self.shadow'
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + (self.decay * self.shadow[name])
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self):
        """
        Applys moving average to the parameter.
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]
    
    def restore(self):
        """
        Restores parameters to pre-moving average calculations.
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
            self.backup = {}

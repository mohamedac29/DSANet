from model.DSANet import DSANet



def build_model(model_name,num_classes):
    if model_name == 'DSANet':
        return DSANet(classes=num_classes)
    else:
        raise NotImplementedError
    

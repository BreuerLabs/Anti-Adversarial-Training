from classifiers.abstract_classifier import AbstractClassifier

def apply_TLDMI_defense(config, model:AbstractClassifier):
    
    class TLDMI(model.__class__): # adapted from TL-DMI repo
        
        def __init__(self, config):
            super(TLDMI, self).__init__(config)

            if config.defense.freeze_layers == 0:
                for param in self.feature_extractor.parameters():
                    param.requires_grad = False

            elif config.defense.freeze_layers > 0:
                for i, child in enumerate(self.feature_extractor.children()):
                    if i <= config.defense.freeze_layers:
                        for j, param in enumerate(child.parameters()):
                            param.requires_grad = False
                            num_trainable = sum(p.numel() for p in self.feature_extractor.parameters() if p.requires_grad)
                    else:
                        break
                        

            num_trainable = sum(p.numel() for p in self.feature_extractor.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.feature_extractor.parameters())
            print(f"{total_params - num_trainable} out of {total_params} parameters are frozen")

    tldmi_defended_model = TLDMI(config)
    return tldmi_defended_model
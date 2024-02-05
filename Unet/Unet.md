U-Net: Convolutional Networks for Biomedical Image Segmentation

Tags:
#paperimplementation #computervision #deeplearning #research 

![[Pasted image 20240203213524.png]]

#### Abstract:
- Basically, they provided a training strategy that uses data augmentation techniques such that we can achieve high accuracy very quick even with less amount of data.
- It consists of a contraction path to capture details i.e. it tells "what" and a symmetric expanding path that enables precise localization i.e. it tells "where".
#### Introduction:
- While convolutional networks have already existed for a long time , their success was limited due to the size of the available training sets and the size of the considered networks.
- After AlexNet breakthrough, many deeper and larger networks have been invented and trained.
- so mostly CNN were used in classification tasks, but in most biomedical image processing , we need to have localization( should tell/detect where an object is i.e. output label is supposed to be assigned to each pixel) and also not much training data is available for such biomedical tasks.
- Previous works used [[Sliding window approach for object detection and localization]] which is very slow and low accuracy whilst being resource intensive.




Implementation:
[U-Net A PyTorch Implementation in 60 lines of Code (amaarora.github.io)](https://amaarora.github.io/posts/2020-09-13-unet.html) 

```
class Block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3)
        self.relu  = nn.ReLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3)
    
    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))


class Encoder(nn.Module):
    def __init__(self, chs=(3,64,128,256,512,1024)):
        super().__init__()
        self.enc_blocks = nn.ModuleList([Block(chs[i], chs[i+1]) for i in range(len(chs)-1)])
        self.pool       = nn.MaxPool2d(2)
    
    def forward(self, x):
        ftrs = []
        for block in self.enc_blocks:
            x = block(x)
            ftrs.append(x)
            x = self.pool(x)
        return ftrs


class Decoder(nn.Module):
    def __init__(self, chs=(1024, 512, 256, 128, 64)):
        super().__init__()
        self.chs         = chs
        self.upconvs    = nn.ModuleList([nn.ConvTranspose2d(chs[i], chs[i+1], 2, 2) for i in range(len(chs)-1)])
        self.dec_blocks = nn.ModuleList([Block(chs[i], chs[i+1]) for i in range(len(chs)-1)]) 
        
    def forward(self, x, encoder_features):
        for i in range(len(self.chs)-1):
            x        = self.upconvs[i](x)
            enc_ftrs = self.crop(encoder_features[i], x)
            x        = torch.cat([x, enc_ftrs], dim=1)
            x        = self.dec_blocks[i](x)
        return x
    
    def crop(self, enc_ftrs, x):
        _, _, H, W = x.shape
        enc_ftrs   = torchvision.transforms.CenterCrop([H, W])(enc_ftrs)
        return enc_ftrs


class UNet(nn.Module):
    def __init__(self, enc_chs=(3,64,128,256,512,1024), dec_chs=(1024, 512, 256, 128, 64), num_class=1, retain_dim=False, out_sz=(572,572)):
        super().__init__()
        self.encoder     = Encoder(enc_chs)
        self.decoder     = Decoder(dec_chs)
        self.head        = nn.Conv2d(dec_chs[-1], num_class, 1)
        self.retain_dim  = retain_dim

    def forward(self, x):
        enc_ftrs = self.encoder(x)
        out      = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:])
        out      = self.head(out)
        if self.retain_dim:
            out = F.interpolate(out, out_sz)
        return out
```


- ==Retain_dim = true changes the size of output to the input size using bilinear interpolation(taking average of neighbouring pixels to fill in new spots to enlarge the image)==
- **why even bother resizing the output image using retain_dim?** cause it is easier to calculate the loss if the output image and input image both are of same dimensions. 
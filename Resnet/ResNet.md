##### Link: https://paperswithcode.com/paper/deep-residual-learning-for-image-recognition

Tags:
#paperimplementation #computervision #deeplearning #research

#### Deep Residual Learning for Image Recognition
### Key takeaways:
1.  Resnet stands for residual neural networks and it is a type of [[Convolutional Neural Networks]]. 
2. Resolves the problem of [[Vanishing and Exploding Gradients]] in very deep neural networks
3. Provides us with Deep ResNet that use Residual Connections
4. **Residual Connections:** ResNet incorporates residual connections, which allow for training very deep neural networks and alleviate the vanishing gradient problem.
5.  **Identity Mapping or skip connections**: ResNet uses identity mapping as the residual function, which makes the training process easier by learning the residual mapping rather than the actual mapping.

### Why Resnet? Problem with old deep learning architectures?
- So, basically the more the number of layers, the more  accurate the output is. 
- But when we go for very deep neural networks, then the accuracy of the model saturates or decreases instead of increasing, i.e. the training and testing error increases. This degradation is not due to [[overfitting]] , but due to the [[Vanishing and Exploding Gradients]] problem. 
	- *When deeper networks are able to start converging, a degradation problem has been exposed: with the network depth increasing, accuracy gets saturated (which might be unsurprising) and then degrades rapidly.*
- This is the degradation problem, where as you make deep neural networks, they become harder to train. The accuracy might plateau or even degrade as you add more layers.
- The paper mentions that vanishing gradients have been addressed by adding the intermediate normalization layers (see [[Batch Normalization]] ) and normalized initialization.
	- *This problem, however, has been largely addressed by normalized initialization [23, 8, 36, 12] and intermediate normalization layers [16], which enable networks with tens of layers to start con verging for stochastic gradient descent (SGD) with back propagation*.
-  In this paper, they address the degradation problem by introducing a deep residual learning framework.
### So, how to resolve this problem?
- Adding more layers to a suitably deep model leads to higher training errors. ❌
- The paper presents how architectural changes like residual learning tackle this degradation problem using residual networks. Residual Network adds an identity mapping between the layers. Applying identity mapping to the input will give the output the same as the input. The skip connections directly pass the input to the output, effectively allowing the network to learn an identity function.![[Pasted image 20240201221831.png]]
- **H(x) = F(x) + x**
where,
       **X** is the input to the set of layers
       **F(x)** is the residual function
       **H(x)** is the mapping function from input to output
- Instead of trying to directly learn the underlying mapping from input to output (H(x)), ResNet introduces the idea of learning a residual mapping (F(x)).
- F(x) represents the difference between the desired output (H(x)) and the input (x). So, instead of trying to learn H(x) directly, the network learns to predict the difference between the desired output and the input.
- The hypothesis is that it's easier to optimize the residual mapping than to optimize the original mapping directly. Essentially, it's easier for the network to learn how to adjust the input rather than learning the entire mapping from scratch.
- **Shortcut/Skip Connections**: These are direct connections between layers that skip one or more layers in between. In ResNet, these shortcut connections perform identity mapping, meaning they simply pass the input through to the next layer unchanged.
- In essence, ResNet tackles the degradation problem in deep neural networks by introducing residual learning, where the network learns to predict the difference between the desired output and the input, making it easier to train very deep networks effectively.
- ![[Pasted image 20240202105926.png]]
- ![[Pasted image 20240202105956.png]] Just learn the residual instead of learning everything.
### Here's how it works in simple terms:
1. **Building Blocks** : ResNet is made up of many smaller blocks called "residual blocks." Each block contains a few layers of neural networks.
2. **Shortcut Connections**: These blocks have a special feature called "shortcut connections" or "skip connections." These connections let the information skip one or more layers and jump directly to a deeper layer in the network.
3. **Learning Residuals**: Instead of trying to directly learn how to map inputs to outputs, ResNet learns to predict the difference or "residual" between the actual output and the desired output. This makes learning easier because the network only needs to focus on small changes, or adjustments, to the input data rather than trying to learn the entire mapping.
4. **Deeper Networks**: Because of these shortcut connections, ResNet can be made much deeper than traditional neural networks without running into problems like vanishing gradients, where the signal gets weaker as it passes through many layers.
5. **Better Performance**: By using these shortcut connections and learning residual functions, ResNet can achieve better accuracy and performance compared to other neural network architectures, especially when dealing with very deep networks.

- **so it is like supervised learning where the model tries to calculate the error or difference between the actual label and predicted value?**
	- In the case of ResNet, it also learns from labeled examples, but instead of directly predicting the output, it learns to predict the difference (residual) between the actual output and the desired output. By focusing on learning these residuals, ResNet can make smaller adjustments to the input data, which can make training easier and more effective, especially in very deep networks.
	
- **and the goal is to minimize the residual?**
	- Exactly! In ResNet, the goal is to minimize the residual, which represents the difference between the actual output and the desired output. By minimizing this residual, the model aims to make its predictions as close to the true labels as possible. This process of minimizing the residual helps the model learn and improve its performance on the task it's trained for, such as image recognition or object detection.
	
- **then how does it resolve the issue of vanishing gradients?**
	- through the use of shortcut connections, also known as skip connections or identity mapping.
	- so we add the input value to the output of residual layers so that we don't lose the gradient?
		- Yes, that's correct! In ResNet, the input to a residual block is added to the output of the block, which is then passed through a non-linear activation function (usually ReLU). This addition operation creates a shortcut connection, also known as a skip connection or identity mapping.
		- By adding the input to the output, the gradient flowing through the network during backpropagation is preserved. Even if the gradient becomes very small in the layers of the residual block, it can still flow through the shortcut connection directly to deeper layers without being significantly affected.

## The ResNet block

![[Pasted image 20240202105422.png]] ![[Pasted image 20240202105806.png]]
```
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1, downsample = None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU())
        self.conv2 = nn.Sequential(
                        nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1),
                        nn.BatchNorm2d(out_channels))
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out
```
- [[Downsample]]  is a parameter for the skip connection because we are summing the original image with the output of conv layers which might have its dimensions reduced.  So, to make the dimensions of both same for summing operation , we need to [[Downsample]] the initial image.
- [[Convolutional Neural Networks]]

*The convolutional layers mostly have 3 x 3 filters and follow two simple design rules: (i) for the same output feature map size, the layers have the same number of filters; and (ii) if the feature map size is halved, the number of filters is doubled so as to preserve the time complexity per layer.*
1. **Consistent Number of Filters**: **For convolutional layers that produce feature maps of the same size, they typically have the same number of filters**. This helps maintain consistency in the information processing across different layers.
	    
2. **Doubling Filters with Halved Feature Map Size**: **When the spatial dimensions of the feature maps are halved (e.g., through operations like max pooling or convolution with a stride of 2), the number of filters in the subsequent convolutional layer is usually doubled.** This increase in filters helps to maintain the computational complexity per layer approximately constant despite the reduction in spatial dimensions. 


![[Pasted image 20240203114332.png]]
The image provided will be 3x224x224 so the first 7x7 conv layer will get 224 size image and it should reduce to 64 output size right?
	- yes that's right. 
	- Output Size: 64x112x112 (64 channels, 112 height, 112 width)
	- number of feature maps will increase from 3 to 64 according to design rule 2.
	- So, after the 7x7 convolutional layer with a stride of 2 and padding of 3, the size of the feature map is indeed reduced from 224x224 to 112x112, but the number of channels increases to 64. 
	- **how to know how many feature maps will be created then? ** 
		- In the ResNet architecture you provided, the first convolutional layer (`nn.Conv2d`) has 64 output channels specified. Therefore, after applying this convolutional layer to the input image of size 224x224, the output will be a feature map with dimensions 64x112x112.


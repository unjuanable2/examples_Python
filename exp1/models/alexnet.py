import torch
import torch.nn as nn
import os
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont, ImageFilter

NUM_CLASSES = 10

tran = transforms.Compose([
    transforms.Resize((32,32), interpolation = Image.BICUBIC),
    transforms.ToTensor(), #Tensor,張量，高維數組
    transforms.Normalize([0.4914, 0.4822, 0.4465],[0.2023, 0.1994, 0.201])
    ])

class AlexNet(nn.Module):
    def __init__(self, num_classes = NUM_CLASSES):
        super(AlexNet, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True), #y=x+1, x=x+1
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 192, kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(192, 384,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256*2*2, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256*2*2)
        x = self.classifier(x)
        return x

def test():
    net = AlexNet().cuda()
    model_path = os.path.join("weights", "alexnet.pt")
    print("Model PATH: " + model_path)
    
    checkpoint = torch.load(model_path)
    net.load_state_dict(checkpoint['net'])

    test_image = os.path.join('test.jpg')
    img = Image.open(test_image)
    img_tensor = tran(img) #CHW, NCHW
    #print(img_tensor.shape)

    input_tensor = img_tensor.unsqueeze_(0).cuda()
    #print(input_tensor.shape)

    y = net(input_tensor)

    #print(y)
    percentage = torch.softmax(y[0], dim=0) * 100
    print('cat percentage:')
    print(percentage)
    cl_fp32, index_fp32 = torch.max(percentage, 0)

    classes = ['plane','car','bird','cat','deer','dog','frog','horse', 'ship','truck']
   
    font = ImageFont.truetype('LiberationSans-Regular.ttf', 30)

    draw = ImageDraw.Draw(img)
    text = str(classes[index_fp32]) + ' (' + '{:.2f}'.format(cl_fp32.item()) + '%' + ')'
    draw.text((0,0), text, font=font, fill="#ff00ff", spacing=0, align='left') 

    img.save(test_image,'jpeg')

if __name__ == '__main__':
    test()
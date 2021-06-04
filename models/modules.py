# https://docs.opensource.microsoft.com/content/releasing/copyright-headers.html
import torch
import functools
import torch.nn as nn
import torch.nn.functional as F




class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        # weight and bias are dynamically assigned
        self.weight = None
        self.bias = None
        # just dummy buffers, not used
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        assert self.weight is not None and self.bias is not None, "Please assign weight and bias before calling AdaIN!"
        b, c = x.size(0), x.size(1)
        running_mean = self.running_mean.repeat(b)   
        running_var = self.running_var.repeat(b)

        # Apply instance norm
        x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])

        out = F.batch_norm(
            x_reshaped, running_mean, running_var, self.weight, self.bias,
            True, self.momentum, self.eps)

        return out.view(b, c, *x.size()[2:])

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'

class ResBlocks(nn.Module):
    def __init__(self, num_blocks, dim, norm='in', activation='relu', pad_type='zero'):
        super(ResBlocks, self).__init__()
        self.model = []
        # num_blocks=4
        # 
        for i in range(num_blocks):
            self.model += [ResBlock(dim, norm=norm, activation=activation, pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)


class ResBlock(nn.Module):
    def __init__(self, dim, norm='in', activation='relu', pad_type='zero'):
        super(ResBlock, self).__init__()

        # initialize padding
        padding = 1
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        self.conv1 = nn.Conv2d(dim, dim, 3, 1, bias=True)
        #print("res_norm:", norm)
        if norm=="in":
            self.norm1 = nn.InstanceNorm2d(dim)
        elif norm=='adain':
            self.norm1 = AdaptiveInstanceNorm2d(dim)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)

        self.conv2 = nn.Conv2d(dim, dim, 3, 1, bias=True)
        if norm=="in":
            self.norm2 = nn.InstanceNorm2d(dim)
        elif norm=='adain':
            self.norm2 = AdaptiveInstanceNorm2d(dim)

    def forward(self, x):
        residual = x

        x = self.conv1(self.pad(x))
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.conv2(self.pad(x))
        out = self.norm2(x)

        out += residual
        return out

def get_nonlinearity_layer(activation_type='PReLU'):
    if activation_type == 'ReLU':
        nonlinearity_layer = nn.ReLU(True)
    elif activation_type == 'SELU':
        nonlinearity_layer = nn.SELU(True)
    elif activation_type == 'LeakyReLU':
        nonlinearity_layer = nn.LeakyReLU(0.1, True)
    elif activation_type == 'PReLU':
        nonlinearity_layer = nn.PReLU()
    else:
        raise NotImplementedError('activation layer [%s] is not found' % activation_type)
    return nonlinearity_layer

class Conv2dBlock(nn.Module):
    # input_dim=3, dim=64, 7, 1, 3, none, lrelu, reflect
    def __init__(self, input_dim ,output_dim, kernel_size, stride,
                 padding=0, norm='none', activation='relu', pad_type='zero'):
        super(Conv2dBlock, self).__init__()
        self.use_bias = True
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim                # 64
        #print("norm_dim:", norm_dim)
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            #self.norm = nn.InstanceNorm2d(norm_dim, track_running_stats=True)
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'adain':
            self.norm = AdaptiveInstanceNorm2d(norm_dim)
        elif norm == 'none' or norm == 'sn':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        if norm == 'sn':
            self.conv = SpectralNorm(nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias))
        else:
            self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias)

    def forward(self, x):
        x = self.pad(x)
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x

class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        # print(x.size())
        if x.size(0) == 1:
            # These two lines run much faster in pytorch 0.4 than the two lines listed below.
            mean = x.view(-1).mean().view(*shape)
            std = x.view(-1).std().view(*shape)
        else:
            mean = x.view(x.size(0), -1).mean(1).view(*shape)
            std = x.view(x.size(0), -1).std(1).view(*shape)

        x = (x - mean) / (std + self.eps)

        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x



class Struct_Encoder(nn.Module):
    def __init__(self, n_downsample, n_res, input_dim, dim, norm, activ, pad_type):
        super(Struct_Encoder, self).__init__()
        self.conv = Conv2dBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type)
        self.model = []
        self.conv1 = Conv2dBlock(dim, 2*dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)
        dim *= 2
        self.conv2 =  Conv2dBlock(dim, 2*dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)
        dim *= 2
        self.resblock = ResBlocks(n_res, dim, norm=norm, activation=activ, pad_type=pad_type)
        self.output_dim = dim

    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.conv1(x1)
        x3 = self.conv2(x2)
        x4 = self.resblock(x3)
        return  x4


class _UpProjection(nn.Sequential):

    def __init__(self, num_input_features, num_output_features):
        super(_UpProjection, self).__init__()

        self.conv1 = nn.Conv2d(num_input_features, num_output_features,
                               kernel_size=5, stride=1, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(num_output_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(num_output_features, num_output_features,
                                 kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1_2 = nn.BatchNorm2d(num_output_features)

        self.conv2 = nn.Conv2d(num_input_features, num_output_features,
                               kernel_size=5, stride=1, padding=2, bias=False)
        self.bn2 = nn.BatchNorm2d(num_output_features)

    def forward(self, x, size):
        x = F.interpolate(x, size=size, mode='bilinear',align_corners=True)
        #x = F.upsample(x, size=size, mode='bilinear')
        x_conv1 = self.relu(self.bn1(self.conv1(x)))
        bran1 = self.bn1_2(self.conv1_2(x_conv1))
        bran2 = self.bn2(self.conv2(x))

        out = self.relu(bran1 + bran2)

        return out


class _EncoderBlock(nn.Module):
    def __init__(self, input_nc, middle_nc, output_nc, norm_layer=nn.BatchNorm2d, nonlinearity=nn.PReLU(), use_bias=False):
        super(_EncoderBlock, self).__init__()

        model = [
            nn.Conv2d(input_nc, middle_nc, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(middle_nc),
            nonlinearity,
            nn.Conv2d(middle_nc, output_nc, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(output_nc),
            nonlinearity
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x) 

class _InceptionBlock(nn.Module):
    def __init__(self, input_nc, output_nc, norm_layer=nn.BatchNorm2d, nonlinearity=nn.PReLU(), width=1, drop_rate=0, use_bias=False):
        super(_InceptionBlock, self).__init__()

        self.width = width
        self.drop_rate = drop_rate

        for i in range(width):     # 0, 1, 2
            layer = nn.Sequential(
                nn.ReflectionPad2d(i*2+1), 
                nn.Conv2d(input_nc, output_nc, kernel_size=3, padding=0, dilation=i*2+1, bias=use_bias)
                )
            setattr(self, 'layer'+str(i), layer)

        self.norm1 = norm_layer(output_nc*width)
        self.norm2 = norm_layer(output_nc)
        self.nonlinearity = nonlinearity
        self.branch1x1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(output_nc*width, output_nc, kernel_size=3, padding=0, bias=use_bias)
            )
    def forward(self, x):
        result = []

        for i in range(self.width):
            layer = getattr(self, 'layer'+str(i))
            result.append(layer(x))
        output = torch.cat(result, 1)
        output = self.nonlinearity(self.norm1(output))
        output = self.norm2(self.branch1x1(output))
        if self.drop_rate > 0:
            output = F.dropout(output, p=self.drop_rate, training=self.training)

        return self.nonlinearity(output+x)

class GaussianNoiseLayer(nn.Module):
    def __init__(self):
        super(GaussianNoiseLayer, self).__init__()

    def forward(self, x):
        if self.training == False:
            return x
        noise = Variable((torch.randn(x.size()).cuda(x.data.get_device()) - 0.5) / 10.0)
        return x+noise


class _DecoderUpBlock(nn.Module):
    def __init__(self, input_nc, middle_nc, output_nc, norm_layer=nn.BatchNorm2d, nonlinearity=nn.PReLU(), use_bias=False):
        super(_DecoderUpBlock, self).__init__()

        model = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(input_nc, middle_nc, kernel_size=3, stride=1, padding=0, bias=use_bias),
            norm_layer(middle_nc),
            nonlinearity,
            nn.ConvTranspose2d(middle_nc, output_nc, kernel_size=3, stride=2, padding=1, output_padding=1),
            norm_layer(output_nc),
            nonlinearity
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

class _OutputBlock(nn.Module):
    def __init__(self, input_nc, output_nc, kernel_size=3, use_bias=False):
        super(_OutputBlock, self).__init__()
        model = [
            nn.ReflectionPad2d(int(kernel_size/2)),
                nn.Conv2d(input_nc, output_nc, kernel_size=kernel_size, padding=0, bias=use_bias),
                nn.Tanh()
            ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class Struct_Decoder(nn.Module):
    def __init__(self):
        super(Struct_Decoder, self).__init__()
        num_features = 256

        self.up1 = _UpProjection(num_input_features=num_features, num_output_features=num_features // 2)
        num_features = num_features // 2           # 128

        self.up2 = _UpProjection(num_input_features=num_features, num_output_features=num_features // 2)
        num_features = num_features // 2           # 64

        self.conv = nn.Conv2d(num_features, 1, kernel_size=3, stride=1, bias=True)
        self.reflect_pad = nn.ReflectionPad2d(1)


    def forward(self, x):
        u1 = self.up1(x, (x.size(2)*2, x.size(3)*2))
        u2 = self.up2(u1, (x.size(2)*4, x.size(3)*4))
        u3 = self.reflect_pad(u2)
        out = torch.sigmoid(self.conv(u3))
        return out

def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=True)
    elif norm_layer == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found'% norm_type)
    return norm_layer


class Depth_Net(nn.Module):
    def __init__(self, input_nc=1, output_nc=1, ngf=64, layers=4, norm='batch', drop_rate=0, add_noise=False, weight=0.1):
        super(Depth_Net, self).__init__()

        self.layers = layers
        self.weight = weight
        norm_layer = get_norm_layer(norm_type=norm)
        nonlinearity = get_nonlinearity_layer(activation_type='PReLU')

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        # encoder part 
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv1 = nn.Sequential(nn.ReflectionPad2d(3),
                                   nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                                   norm_layer(ngf),
                                   nonlinearity)
        self.conv2 = _EncoderBlock(ngf, ngf*2, ngf*2, norm_layer, nonlinearity, use_bias)    # 64/128/128
        self.conv3 = _EncoderBlock(ngf*2, ngf*4, ngf*4, norm_layer, nonlinearity, use_bias)  # 128/256/256
        self.conv4 = _EncoderBlock(ngf*4, ngf*8, ngf*8, norm_layer, nonlinearity, use_bias)  # 256/512/512

        for i in range(layers - 4):
            conv = _EncoderBlock(ngf*8, ngf*8, ngf*8, norm_layer, nonlinearity, use_bias)
            setattr(self, 'down'+str(i), conv.model)

        center = []
        for i in range(7 - layers):   # 0, 1, 2
            center += [
                    _InceptionBlock(ngf*8, ngf*8, norm_layer, nonlinearity, 7-layers, drop_rate, use_bias)
                ]

        center += [
        _DecoderUpBlock(ngf*8, ngf*8, ngf*4, norm_layer, nonlinearity, use_bias)
        ]

        if add_noise:
            center += [GaussianNoiseLayer()]
        self.center = nn.Sequential(*center)

        for i in range(layers-4):
            upconv = _DecoderUpBlock(ngf*(8+4), ngf*8, ngf*4, norm_layer, nonlinearity, use_bias)
            setattr(self, 'up'+str(i), upconv.model)

        self.deconv4 = _DecoderUpBlock(ngf*(4+4), ngf*8, ngf*2, norm_layer, nonlinearity, use_bias)
        self.deconv3 = _DecoderUpBlock(ngf*(2+2)+output_nc, ngf*4, ngf, norm_layer, nonlinearity, use_bias)
        self.deconv2 = _DecoderUpBlock(ngf*(1+1)+output_nc, ngf*2, int(ngf/2), norm_layer, nonlinearity, use_bias)

        self.output4 = _OutputBlock(ngf*(4+4), output_nc, 3, use_bias)
        self.output3 = _OutputBlock(ngf*(2+2)+output_nc, output_nc, 3, use_bias)
        self.output2 = _OutputBlock(ngf*(1+1)+output_nc, output_nc, 3, use_bias)
        self.output1 = _OutputBlock(int(ngf/2)+output_nc, output_nc, 7, use_bias)

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, input):
        conv1 = self.pool(self.conv1(input))          # 3/64 1/2
        conv2 = self.pool(self.conv2.forward(conv1))  # 64/128 1/4
        conv3 = self.pool(self.conv3.forward(conv2))  # 128/256 1/8
        
        center_in = self.pool(self.conv4.forward(conv3))   # 256/512 1/16

        middle = [center_in]
        for i in range(self.layers-4):
            model = getattr(self, 'down'+str(i))
            center_in = self.pool(model.forward(center_in))
            middle.append(center_in)
        center_out = self.center.forward(center_in)

        for i in range(self.layers-4):
            model = getattr(self, 'up'+str(i))
            center_out = model.forward(torch.cat([center_out, middle[self.layers-5-i]], 1))

        scale = 1.0
        result= []
        deconv4 = self.deconv4.forward(torch.cat([center_out, conv3 * self.weight], 1))
        output4 = scale * self.output4.forward(torch.cat([center_out, conv3 * self.weight], 1))
        result.append(output4)
        deconv3 = self.deconv3.forward(torch.cat([deconv4, conv2 * self.weight * 0.5, self.upsample(output4)], 1))
        output3 = scale * self.output3.forward(torch.cat([deconv4, conv2 * self.weight * 0.5, self.upsample(output4)], 1))
        result.append(output3)
        deconv2 = self.deconv2.forward(torch.cat([deconv3, conv1 * self.weight * 0.1, self.upsample(output3)], 1))
        output2 = scale * self.output2.forward(torch.cat([deconv3, conv1 * self.weight * 0.1, self.upsample(output3)], 1))
        result.append(output2)
        output1 = scale * self.output1.forward(torch.cat([deconv2, self.upsample(output2)], 1))
        result.append(output1)

        return result
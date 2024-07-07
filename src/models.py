

#DNN
class AIGCLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3,5), stride=(1,1)):
        super(AIGCLayer, self).__init__()

        self.conv_re = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, bias=False, padding='same')
        self.conv_im = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, bias=False, padding='same')
        self.conv_gate = nn.Conv2d(in_channels+1, out_channels, kernel_size, stride=stride, bias=False, padding='same')

    def forward(self, x_re, x_im, amplitude):
        h_re = self.conv_re(x_re) - self.conv_im(x_im)
        h_im = self.conv_re(x_im) + self.conv_im(x_re)
        h_gate = self.conv_gate(torch.cat([torch.abs(torch.complex(x_re, x_im)), amplitude], dim=1))

        f_re = h_re * F.sigmoid(h_gate)
        f_im = h_im * F.sigmoid(h_gate)
        return f_re, f_im, F.sigmoid(h_gate)



class AIGCNN(nn.Module):
    def __init__(self, num_ch, kernel_size=(3,5), stride=(1,1)):
        super(AIGCNN, self).__init__()

        self.layer1 = AIGCLayer(3, num_ch, kernel_size=kernel_size, stride=stride)

        self.layer2 = AIGCLayer(num_ch, num_ch, kernel_size=kernel_size, stride=stride)

        self.layer3 = AIGCLayer(num_ch, num_ch, kernel_size=kernel_size, stride=stride)

        self.conv_re = nn.Conv2d(num_ch, 1, kernel_size=(1,1), stride=stride, bias=False, padding='same')
        self.conv_im = nn.Conv2d(num_ch, 1, kernel_size=(1,1), stride=stride, bias=False, padding='same')
                        
    def forward(self, x_re, x_im, amp):
        ampg = amp.unsqueeze(1)

        f_re, f_im, gate1 = self.layer1(x_re, x_im, ampg)
        f_re, f_im, gate2 = self.layer2(f_re, f_im, ampg)
        f_re, f_im, gate3 = self.layer3(f_re, f_im, ampg)
        
        out_re = self.conv_re(f_re) - self.conv_im(f_im)
        out_im = self.conv_re(f_im) + self.conv_im(f_re)
        out = torch.complex(out_re, out_im).squeeze()        
        return out, (gate1, gate2, gate3)
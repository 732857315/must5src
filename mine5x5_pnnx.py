# pnnx model stat
# model inputshape = [1,1,5,5]f32
# FLOPS = 3.785M
# memory OPS = 124.711K

import os
import numpy as np
import tempfile, zipfile
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    import torchvision
    import torchaudio
except:
    pass

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.conv2d_8 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=1, kernel_size=(3,3), out_channels=32, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.conv2d_9 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=32, kernel_size=(3,3), out_channels=32, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.conv2d_10 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=32, kernel_size=(3,3), out_channels=32, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.conv2d_0 = nn.Conv2d(bias=False, dilation=(1,1), groups=1, in_channels=32, kernel_size=(1,1), out_channels=8, padding=(0,0), padding_mode='zeros', stride=(1,1))
        self.conv2d_1 = nn.Conv2d(bias=False, dilation=(1,1), groups=1, in_channels=8, kernel_size=(1,1), out_channels=32, padding=(0,0), padding_mode='zeros', stride=(1,1))
        self.conv2d_11 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=32, kernel_size=(3,3), out_channels=32, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.conv2d_12 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=32, kernel_size=(3,3), out_channels=32, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.conv2d_2 = nn.Conv2d(bias=False, dilation=(1,1), groups=1, in_channels=32, kernel_size=(1,1), out_channels=8, padding=(0,0), padding_mode='zeros', stride=(1,1))
        self.conv2d_3 = nn.Conv2d(bias=False, dilation=(1,1), groups=1, in_channels=8, kernel_size=(1,1), out_channels=32, padding=(0,0), padding_mode='zeros', stride=(1,1))
        self.conv2d_13 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=32, kernel_size=(3,3), out_channels=32, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.conv2d_14 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=32, kernel_size=(3,3), out_channels=32, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.conv2d_4 = nn.Conv2d(bias=False, dilation=(1,1), groups=1, in_channels=32, kernel_size=(1,1), out_channels=8, padding=(0,0), padding_mode='zeros', stride=(1,1))
        self.conv2d_5 = nn.Conv2d(bias=False, dilation=(1,1), groups=1, in_channels=8, kernel_size=(1,1), out_channels=32, padding=(0,0), padding_mode='zeros', stride=(1,1))
        self.conv2d_15 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=32, kernel_size=(3,3), out_channels=32, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.conv2d_16 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=32, kernel_size=(3,3), out_channels=32, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.conv2d_6 = nn.Conv2d(bias=False, dilation=(1,1), groups=1, in_channels=32, kernel_size=(1,1), out_channels=8, padding=(0,0), padding_mode='zeros', stride=(1,1))
        self.conv2d_7 = nn.Conv2d(bias=False, dilation=(1,1), groups=1, in_channels=8, kernel_size=(1,1), out_channels=32, padding=(0,0), padding_mode='zeros', stride=(1,1))
        self.conv2d_17 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=32, kernel_size=(1,1), out_channels=16, padding=(0,0), padding_mode='zeros', stride=(1,1))
        self.conv2d_18 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=16, kernel_size=(1,1), out_channels=1, padding=(0,0), padding_mode='zeros', stride=(1,1))
        self.conv2d_19 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=32, kernel_size=(1,1), out_channels=16, padding=(0,0), padding_mode='zeros', stride=(1,1))
        self.F_linear_0 = nn.Linear(bias=True, in_features=16, out_features=32)
        self.F_linear_1 = nn.Linear(bias=True, in_features=32, out_features=1)

        archive = zipfile.ZipFile('mine5x5.pnnx.bin', 'r')
        self.conv2d_8.bias = self.load_pnnx_bin_as_parameter(archive, 'conv2d_8.bias', (32), 'float32')
        self.conv2d_8.weight = self.load_pnnx_bin_as_parameter(archive, 'conv2d_8.weight', (32,1,3,3), 'float32')
        self.conv2d_9.bias = self.load_pnnx_bin_as_parameter(archive, 'conv2d_9.bias', (32), 'float32')
        self.conv2d_9.weight = self.load_pnnx_bin_as_parameter(archive, 'conv2d_9.weight', (32,32,3,3), 'float32')
        self.conv2d_10.bias = self.load_pnnx_bin_as_parameter(archive, 'conv2d_10.bias', (32), 'float32')
        self.conv2d_10.weight = self.load_pnnx_bin_as_parameter(archive, 'conv2d_10.weight', (32,32,3,3), 'float32')
        self.conv2d_0.weight = self.load_pnnx_bin_as_parameter(archive, 'conv2d_0.weight', (8,32,1,1), 'float32')
        self.conv2d_1.weight = self.load_pnnx_bin_as_parameter(archive, 'conv2d_1.weight', (32,8,1,1), 'float32')
        self.conv2d_11.bias = self.load_pnnx_bin_as_parameter(archive, 'conv2d_11.bias', (32), 'float32')
        self.conv2d_11.weight = self.load_pnnx_bin_as_parameter(archive, 'conv2d_11.weight', (32,32,3,3), 'float32')
        self.conv2d_12.bias = self.load_pnnx_bin_as_parameter(archive, 'conv2d_12.bias', (32), 'float32')
        self.conv2d_12.weight = self.load_pnnx_bin_as_parameter(archive, 'conv2d_12.weight', (32,32,3,3), 'float32')
        self.conv2d_2.weight = self.load_pnnx_bin_as_parameter(archive, 'conv2d_2.weight', (8,32,1,1), 'float32')
        self.conv2d_3.weight = self.load_pnnx_bin_as_parameter(archive, 'conv2d_3.weight', (32,8,1,1), 'float32')
        self.conv2d_13.bias = self.load_pnnx_bin_as_parameter(archive, 'conv2d_13.bias', (32), 'float32')
        self.conv2d_13.weight = self.load_pnnx_bin_as_parameter(archive, 'conv2d_13.weight', (32,32,3,3), 'float32')
        self.conv2d_14.bias = self.load_pnnx_bin_as_parameter(archive, 'conv2d_14.bias', (32), 'float32')
        self.conv2d_14.weight = self.load_pnnx_bin_as_parameter(archive, 'conv2d_14.weight', (32,32,3,3), 'float32')
        self.conv2d_4.weight = self.load_pnnx_bin_as_parameter(archive, 'conv2d_4.weight', (8,32,1,1), 'float32')
        self.conv2d_5.weight = self.load_pnnx_bin_as_parameter(archive, 'conv2d_5.weight', (32,8,1,1), 'float32')
        self.conv2d_15.bias = self.load_pnnx_bin_as_parameter(archive, 'conv2d_15.bias', (32), 'float32')
        self.conv2d_15.weight = self.load_pnnx_bin_as_parameter(archive, 'conv2d_15.weight', (32,32,3,3), 'float32')
        self.conv2d_16.bias = self.load_pnnx_bin_as_parameter(archive, 'conv2d_16.bias', (32), 'float32')
        self.conv2d_16.weight = self.load_pnnx_bin_as_parameter(archive, 'conv2d_16.weight', (32,32,3,3), 'float32')
        self.conv2d_6.weight = self.load_pnnx_bin_as_parameter(archive, 'conv2d_6.weight', (8,32,1,1), 'float32')
        self.conv2d_7.weight = self.load_pnnx_bin_as_parameter(archive, 'conv2d_7.weight', (32,8,1,1), 'float32')
        self.conv2d_17.bias = self.load_pnnx_bin_as_parameter(archive, 'conv2d_17.bias', (16), 'float32')
        self.conv2d_17.weight = self.load_pnnx_bin_as_parameter(archive, 'conv2d_17.weight', (16,32,1,1), 'float32')
        self.conv2d_18.bias = self.load_pnnx_bin_as_parameter(archive, 'conv2d_18.bias', (1), 'float32')
        self.conv2d_18.weight = self.load_pnnx_bin_as_parameter(archive, 'conv2d_18.weight', (1,16,1,1), 'float32')
        self.conv2d_19.bias = self.load_pnnx_bin_as_parameter(archive, 'conv2d_19.bias', (16), 'float32')
        self.conv2d_19.weight = self.load_pnnx_bin_as_parameter(archive, 'conv2d_19.weight', (16,32,1,1), 'float32')
        self.F_linear_0.bias = self.load_pnnx_bin_as_parameter(archive, 'F_linear_0.bias', (32), 'float32')
        self.F_linear_0.weight = self.load_pnnx_bin_as_parameter(archive, 'F_linear_0.weight', (32,16), 'float32')
        self.F_linear_1.bias = self.load_pnnx_bin_as_parameter(archive, 'F_linear_1.bias', (1), 'float32')
        self.F_linear_1.weight = self.load_pnnx_bin_as_parameter(archive, 'F_linear_1.weight', (1,32), 'float32')
        archive.close()

    def load_pnnx_bin_as_parameter(self, archive, key, shape, dtype, requires_grad=True):
        return nn.Parameter(self.load_pnnx_bin_as_tensor(archive, key, shape, dtype), requires_grad)

    def load_pnnx_bin_as_tensor(self, archive, key, shape, dtype):
        fd, tmppath = tempfile.mkstemp()
        with os.fdopen(fd, 'wb') as tmpf, archive.open(key) as keyfile:
            tmpf.write(keyfile.read())
        m = np.memmap(tmppath, dtype=dtype, mode='r', shape=shape).copy()
        os.remove(tmppath)
        return torch.from_numpy(m)

    def forward(self, v_0):
        v_1 = self.conv2d_8(v_0)
        v_2 = F.relu(v_1)
        v_3 = self.conv2d_9(v_2)
        v_4 = F.relu(v_3)
        v_5 = self.conv2d_10(v_4)
        v_6 = torch.mean(v_5, dim=(-1,-2), keepdim=True)
        v_7 = self.conv2d_0(v_6)
        v_8 = F.relu(v_7)
        v_9 = self.conv2d_1(v_8)
        v_10 = F.sigmoid(v_9)
        v_11 = ((v_5 * v_10) + v_2)
        v_12 = F.relu(v_11)
        v_13 = self.conv2d_11(v_12)
        v_14 = F.relu(v_13)
        v_15 = self.conv2d_12(v_14)
        v_16 = torch.mean(v_15, dim=(-1,-2), keepdim=True)
        v_17 = self.conv2d_2(v_16)
        v_18 = F.relu(v_17)
        v_19 = self.conv2d_3(v_18)
        v_20 = F.sigmoid(v_19)
        v_21 = ((v_15 * v_20) + v_12)
        v_22 = F.relu(v_21)
        v_23 = self.conv2d_13(v_22)
        v_24 = F.relu(v_23)
        v_25 = self.conv2d_14(v_24)
        v_26 = torch.mean(v_25, dim=(-1,-2), keepdim=True)
        v_27 = self.conv2d_4(v_26)
        v_28 = F.relu(v_27)
        v_29 = self.conv2d_5(v_28)
        v_30 = F.sigmoid(v_29)
        v_31 = ((v_25 * v_30) + v_22)
        v_32 = F.relu(v_31)
        v_33 = self.conv2d_15(v_32)
        v_34 = F.relu(v_33)
        v_35 = self.conv2d_16(v_34)
        v_36 = torch.mean(v_35, dim=(-1,-2), keepdim=True)
        v_37 = self.conv2d_6(v_36)
        v_38 = F.relu(v_37)
        v_39 = self.conv2d_7(v_38)
        v_40 = F.sigmoid(v_39)
        v_41 = ((v_35 * v_40) + v_32)
        v_42 = F.relu(v_41)
        v_43 = self.conv2d_17(v_42)
        v_44 = F.relu(v_43)
        v_45 = self.conv2d_18(v_44)
        v_46 = v_45.reshape(1, 25)
        v_47 = self.conv2d_19(v_42)
        v_48 = F.relu(v_47)
        v_49 = torch.mean(v_48, dim=(-1,-2), keepdim=True)
        v_50 = v_49.reshape(1, 16)
        v_51 = self.F_linear_0(v_50)
        v_52 = F.relu(v_51)
        v_53 = self.F_linear_1(v_52)
        v_54 = F.tanh(v_53)
        v_55 = torch.squeeze(v_54, dim=-1)
        return v_46, v_55

def export_torchscript():
    net = Model()
    net.float()
    net.eval()

    torch.manual_seed(0)
    v_0 = torch.rand(1, 1, 5, 5, dtype=torch.float)

    mod = torch.jit.trace(net, v_0)
    mod.save("mine5x5_pnnx.py.pt")

def export_onnx():
    net = Model()
    net.float()
    net.eval()

    torch.manual_seed(0)
    v_0 = torch.rand(1, 1, 5, 5, dtype=torch.float)

    torch.onnx.export(net, v_0, "mine5x5_pnnx.py.onnx", export_params=True, operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK, opset_version=13, input_names=['in0'], output_names=['out0', 'out1'])

def export_pnnx():
    net = Model()
    net.float()
    net.eval()

    torch.manual_seed(0)
    v_0 = torch.rand(1, 1, 5, 5, dtype=torch.float)

    import pnnx
    pnnx.export(net, "mine5x5_pnnx.py.pt", v_0)

def export_ncnn():
    export_pnnx()

@torch.no_grad()
def test_inference():
    net = Model()
    net.float()
    net.eval()

    torch.manual_seed(0)
    v_0 = torch.rand(1, 1, 5, 5, dtype=torch.float)

    return net(v_0)

if __name__ == "__main__":
    print(test_inference())

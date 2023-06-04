import torch
import torch.nn as nn
from torch import optim
import time
import os
from PIL import Image
from torchvision import transforms as T
from torchvision import models
from torch.utils.tensorboard import SummaryWriter
import config as cfg
import sys


class StyleNetwork(nn.Module):
	def __init__(self, loadpath=None):
		super(StyleNetwork, self).__init__()
		self.loadpath=loadpath

		self.layer1 = self.get_conv_module(inc=3, outc=16, ksize=9)

		self.layer2 = self.get_conv_module(inc=16, outc=32)

		self.layer3 = self.get_conv_module(inc=32, outc=64)

		self.layer4 = self.get_conv_module(inc=64, outc=128)

		self.connector1=self.get_depthwise_separable_module(128, 128)

		self.connector2=self.get_depthwise_separable_module(64, 64)

		self.connector3=self.get_depthwise_separable_module(32, 32)

		self.layer5 = self.get_deconv_module(256, 64)

		self.layer6 = self.get_deconv_module(128, 32)

		self.layer7 = self.get_deconv_module(64, 16)

		self.layer8 = nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1)

		self.activation=nn.Sigmoid()

		if self.loadpath:
			self.load_state_dict(torch.load(self.loadpath))

	def get_conv_module(self, inc, outc, ksize=3):
		padding=(ksize-1)//2
		conv=nn.Conv2d(in_channels=inc, out_channels=outc, kernel_size=ksize, stride=2, padding=padding)
		bn=nn.BatchNorm2d(outc)
		relu=nn.LeakyReLU(0.1)

		return nn.Sequential(conv, bn, relu)

	def get_deconv_module(self, inc, outc, ksize=3):
		padding=(ksize-1)//2
		tconv=nn.ConvTranspose2d(inc, outc, kernel_size=ksize, stride=2, padding=padding, output_padding=padding)
		bn=nn.BatchNorm2d(outc)
		relu=nn.LeakyReLU(0.1)

		return nn.Sequential(tconv, bn, relu)


	def get_depthwise_separable_module(self, inc, outc):
		"""
		inc(int): number of input channels
		outc(int): number of output channels

		Implements a depthwise separable convolution layer
		along with batch norm and activation.
		Intended to be used with inc=outc in the current architecture
		"""
		depthwise=nn.Conv2d(inc, inc, kernel_size=3, stride=1, padding=1, groups=inc)
		pointwise=nn.Conv2d(inc, outc, kernel_size=1, stride=1, padding=0, groups=1)
		bn_layer=nn.BatchNorm2d(outc)
		activation=nn.LeakyReLU(0.1)

		return nn.Sequential(depthwise, pointwise, bn_layer, activation)

	def forward(self, x):

		x=self.layer1(x)

		x2=self.layer2(x)

		x3=self.layer3(x2)

		x4=self.layer4(x3)

		xs4=self.connector1(x4)
		xs3=self.connector2(x3)
		xs2=self.connector3(x2)

		c1=torch.cat([x4, xs4], dim=1)

		x5=self.layer5(c1)

		c2=torch.cat([x5, xs3], dim=1)

		x6=self.layer6(c2)

		c3=torch.cat([x6, xs2], dim=1)

		x7=self.layer7(c3)

		out=self.layer8(x7)

		out=self.activation(out)

		return out


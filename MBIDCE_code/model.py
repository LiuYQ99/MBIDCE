import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class enhance_net_nopool(nn.Module):
	def __init__(self):
		super(enhance_net_nopool, self).__init__()
		number_f = 8
		self.e_conv1 = nn.Sequential(
			nn.Conv2d(3, number_f, 3, 1, 1, bias=True),
			nn.ReLU(inplace=True),
		)
		self.e_conv2 = nn.Sequential(
			nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True),
			nn.ReLU(inplace=True),
		)
		self.e_conv3 = nn.Sequential(
			nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True),
			nn.ReLU(inplace=True),
		)
		self.e_conv4 = nn.Sequential(
			nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True),
			nn.ReLU(inplace=True),
		)
		self.e_conv5 = nn.Sequential(
			nn.Conv2d(number_f * 2, number_f * 1, 1, 1, bias=True),
			nn.ReLU(inplace=True),
			nn.Conv2d(number_f * 1, number_f * 1, 3, 1, 1, bias=True),
			nn.ReLU(inplace=True),
			nn.Conv2d(number_f * 1, number_f * 2, 1, 1, bias=True),
			# nn.ReLU(inplace=True),
		)
		self.e_conv5_out = nn.Sequential(
			nn.Conv2d(number_f * 2, number_f, 1, 1, bias=True),
			nn.ReLU(inplace=True),
			nn.Conv2d(number_f * 1, number_f, 3, 1, 1, bias=True),
			nn.ReLU(inplace=True),
			nn.Conv2d(number_f, 3, 1, 1, bias=True),
		)
		self.e_conv6 = nn.Sequential(
			nn.Conv2d(number_f * 3, number_f * 1, 1, 1, bias=True),
			nn.ReLU(inplace=True),
			nn.Conv2d(number_f * 1, number_f * 1, 3, 1, 1, bias=True),
			nn.ReLU(inplace=True),
			nn.Conv2d(number_f * 1, number_f * 3, 1, 1, bias=True),
			# nn.ReLU(inplace=True),
		)
		self.e_conv6_out = nn.Sequential(
			nn.Conv2d(number_f * 3, number_f, 1, 1, bias=True),
			nn.ReLU(inplace=True),
			nn.Conv2d(number_f * 1, number_f, 3, 1, 1, bias=True),
			nn.ReLU(inplace=True),
			nn.Conv2d(number_f, 3, 1, 1, bias=True),
		)
		self.e_conv7 = nn.Sequential(
			nn.Conv2d(number_f * 4, number_f, 1, 1, bias=True),
			# nn.Conv2d(number_f * 4, 3, 1, 1, bias=True),
			nn.ReLU(inplace=True),
			nn.Conv2d(number_f * 1, number_f, 3, 1, 1, bias=True),
			nn.ReLU(inplace=True),
			nn.Conv2d(number_f, 3, 1, 1, bias=True),
		)
		self.relu = nn.ReLU(inplace=True)

	def forward(self, x):

		x1 = self.e_conv1(x)

		x2 = self.e_conv2(x1)

		x3 = self.e_conv3(x2)

		x4 = self.e_conv4(x3)

		x5 = self.relu(self.e_conv5(torch.cat([x3,x4],1)) + torch.cat([x3,x4],1))

		x5_out = torch.tanh(self.e_conv5_out(x5))
		x = x + x5_out * x * (x-1) * (x - 2)
		# x_1 = x

		x6 = self.relu(self.e_conv6(torch.cat([x2,x5],1)) + torch.cat([x2,x5],1))

		x6_out = torch.tanh(self.e_conv6_out(x6))
		x = x + x6_out * x * (x-1) * (x - 2)
		# x_2 = x

		x7 = torch.tanh(self.e_conv7(torch.cat([x1, x6], 1)))
		enhanced = x + x7 * x * (x-1) * (x - 2)

		x7 = torch.cat([x5_out,x6_out,x7],1)

		return enhanced, x7




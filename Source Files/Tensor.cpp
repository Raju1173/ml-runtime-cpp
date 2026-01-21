#include "Tensor.h"
#include <cstdlib>
#include <iostream>

Tensor::Tensor(const std::vector<int>& shape) : shape(shape), pData(nullptr), numel(1)
{
	for (int dim : shape)
	{
		numel *= dim;
	}

	pData = (float*)std::malloc(numel * sizeof(float));
}

Tensor::~Tensor()
{
	free(pData);
}

void add(const Tensor& a, const Tensor& b, Tensor& out)
{
	if (a.numel != b.numel)
		throw std::runtime_error("add : numel mismatch");

	for (size_t i = 0; i < a.numel; i++)
	{
		out.pData[i] = a.pData[i] + b.pData[i];
	}
}

void matmul(const Tensor& a, const Tensor& b, Tensor& out)
{
	if (a.shape.size() != 2 || b.shape.size() != 2) 
	{
		throw std::runtime_error("matmul : only 2D tensors supported");
	}

	size_t m = a.shape[0];
	size_t kA = a.shape[1];
	size_t kB = b.shape[0];
	size_t n = b.shape[1];

	if (kA != kB)
	{
		throw std::runtime_error("matmul: shape mismatch");
	}

	for (size_t i = 0; i < m; i++)
	{
		for (size_t j = 0; j < n; j++)
		{
			float sum = 0;

			for (size_t t = 0; t < kA; t++)
			{
				sum += a.pData[i * kA + t] * b.pData[t * n + j];
			}

			out.pData[i * n + j] = sum;
		}
	}
}

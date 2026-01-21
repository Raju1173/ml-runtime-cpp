#pragma once
#include<vector>
#include<cstddef>

struct Tensor 
{
	float* pData;
	std::vector<int> shape;
	size_t numel;

	Tensor(const std::vector<int>& shape);
	~Tensor();
};

void add(const Tensor& a, const Tensor& b, Tensor& out);
void matmul(const Tensor& a, const Tensor& b, Tensor& out);

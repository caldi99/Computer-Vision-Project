#include "../include/Utils.h"

template <typename T>
std::vector<int>  Utils::argSort(const std::vector<T>& vector) 
{
	std::vector<int> indices(vector.size());

	//Fill indices with as many 0 as vector.size()
	std::iota(indices.begin(), indices.end(), 0);

	// sort indexes based on comparing values in vector using std::stable_sort instead of std::sort to avoid unnecessary index re-orderings when vector contains elements of equal values 
	std::stable_sort(indices.begin(), indices.end(), [&v](int i1, int i2) { return v[i1] < v[i2];  });

	return indices;
}

template<typename T>
std::vector<T> Utils::slice(std::vector<T>& vector, int start, int end)
{
	// Starting and Ending iterators
	auto start = vector.begin() + start;
	auto end = vector.begin() + end + 1;

	// To store the sliced vector
	std::vector<T> result(end - start + 1);

	// Copy vector using copy function()
	std::copy(start, end, result.begin());

	// Return the final sliced vector
	return result;	
}

template<typename T>
std::vector<T> Utils::slice(std::vector<T>& vector, std::vector<int>& indices)
{
	std::vector<T> ret;
	for (int i = 0; i < indices.size(); i++)	
		ret.push_back(vector.at(indices.at(i)));
	return ret;
}

template<typename T>
T Utils::maximum(std::vector<T>& vector, T& element)
{
	T max = element;
	for (int i = 0; i < vector.size(); i++)	
		if (max < vector.at(i))
			max = vector.at(i);
	return max;
}

template<typename T>
std::vector<T> Utils::elementWiseProduct(std::vector<T>& vector1, std::vector<T>& vector2)
{
	if (vector1.size() != vector2.size())
		throw std::exception("VECTORS OF DIFFERENT SIZES");

	std::vector<T> ret;
	for (int i = 0; i < vector1.size(); i++)	
		ret.push_back(vector1.at(i) * vector2.at(i));
	return ret;
}

template<typename T>
std::vector<T> Utils::elementWiseDifference(std::vector<T>& vector1, std::vector<T>& vector2)
{
	if (vector1.size() != vector2.size())
		throw std::exception("VECTORS OF DIFFERENT SIZES");

	std::vector<T> ret;
	for (int i = 0; i < vector1.size(); i++)
		ret.push_back(vector1.at(i) - vector2.at(i));
	return ret;
}

template<typename T>
std::vector<T> Utils::elementWiseSum(std::vector<T>& vector, T& element)
{
	std::vector<T> ret;
	for (int i = 0; i < vector1.size(); i++)
		ret.push_back(vector1.at(i) + element);
	return ret;
}

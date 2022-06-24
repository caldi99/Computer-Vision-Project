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
std::vector<T> Utils::slicing(std::vector<T>& vector, int start, int end)
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

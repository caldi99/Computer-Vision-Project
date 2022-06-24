#include "../include/Utils.h"

template <typename T> std::vector<int>  Utils::argSort(const std::vector<T>& vector) 
{
	std::vector<int> indices(vector.size());

	//Fill indices with as many 0 as vector.size()
	std::iota(indices.begin(), indices.end(), 0);

	// sort indexes based on comparing values in vector using std::stable_sort instead of std::sort to avoid unnecessary index re-orderings when vector contains elements of equal values 
	std::stable_sort(indices.begin(), indices.end(), [&vector](int i1, int i2) { return vector[i1] < vector[i2];  });

	return indices;
}

template<typename T> std::vector<T> Utils::slice(std::vector<T>& vector, int start, int end)
{
	// Starting and Ending iterators
	auto s = vector.begin() + start;
	auto e = vector.begin() + end + 1;

	// To store the sliced vector
	std::vector<T> result(e - s + 1);

	// Copy vector using copy function()
	std::copy(s, e, result.begin());

	// Return the final sliced vector
	return result;	
}

template<typename T> std::vector<T> Utils::slice(std::vector<T>& vector, std::vector<int>& indices)
{
	std::vector<T> ret;
	for (int i = 0; i < indices.size(); i++)	
		ret.push_back(vector.at(indices.at(i)));
	return ret;
}

template<typename T>  std::vector<T> Utils::elementWiseMaximum(std::vector<T>& vector, T element)
{
	std::vector<T> ret;
	for (int i = 0; i < vector.size(); i++)
		if (vector.at(i) < element)
			ret.push_back(element);
		else
			ret.push_back(vector.at(i));
	return ret;
}

template<typename T>  std::vector<T> Utils::elementWiseProduct(std::vector<T>& vector1, std::vector<T>& vector2)
{
	if (vector1.size() != vector2.size())
		throw std::exception("VECTORS OF DIFFERENT SIZES");

	std::vector<T> ret;
	for (int i = 0; i < vector1.size(); i++)	
		ret.push_back(vector1.at(i) * vector2.at(i));
	return ret;
}

template<typename T>  std::vector<T> Utils::elementWiseDifference(std::vector<T>& vector1, std::vector<T>& vector2)
{
	if (vector1.size() != vector2.size())
		throw std::exception("VECTORS OF DIFFERENT SIZES");

	std::vector<T> ret;
	for (int i = 0; i < vector1.size(); i++)
		ret.push_back(vector1.at(i) - vector2.at(i));
	return ret;
}

template<typename T>  std::vector<T> Utils::elementWiseDivision(std::vector<T>& vector1, std::vector<T>& vector2)
{
	if (vector1.size() != vector2.size())
		throw std::exception("VECTORS OF DIFFERENT SIZES");

	std::vector<T> ret;
	for (int i = 0; i < vector1.size(); i++)
		ret.push_back(vector1.at(i) / vector2.at(i));
	return ret;
}

template<typename T>  std::vector<T> Utils::elementWiseSum(std::vector<T>& vector, T element)
{
	std::vector<T> ret;
	for (int i = 0; i < vector.size(); i++)
		ret.push_back(vector.at(i) + element);
	return ret;
}

template<typename T>  std::vector<int> Utils::greater(std::vector<T>& vector, T threshold)
{
	std::vector<int> ret;
	for (int i = 0; i < vector.size(); i++)
		if (vector.at(i) > threshold)
			ret.push_back(i);
	return ret;
}

template<typename T> void Utils::deleteElementPositions(std::vector<T>& vector, std::vector<int>& positions)
{
	
	//sort positions in descending order
	std::sort(positions.begin(), positions.end(), std::greater<int>());
	positions.erase(std::unique(positions.begin(), positions.end()), positions.end());

	if (vector.size() > positions.size())
		throw std::exception("MORE POSITIONS THAN ELEMENTS");

	for (int i = 0; i < positions.size(); i++)	
		vector.erase(vector.begin() + positions.at(i));	
}

//USAGES : For each specific usage instace add a line of that instance here
template std::vector<int> Utils::argSort(const std::vector<float>&);
template std::vector<int> Utils::slice(std::vector<int>& vector,int,int);
template std::vector<float> Utils::slice(std::vector<float>&, std::vector<int>&);
template std::vector<cv::Rect2f> Utils::slice(std::vector<cv::Rect2f>&, std::vector<int>&);
template std::vector<float> Utils::elementWiseMaximum(std::vector<float>&, float);
template std::vector<float> Utils::elementWiseProduct(std::vector<float>& vector1, std::vector<float>& vector2);
template std::vector<float> Utils::elementWiseDifference(std::vector<float>& vector1, std::vector<float>& vector2);
template std::vector<float> Utils::elementWiseDivision(std::vector<float>& vector1, std::vector<float>& vector2);
template std::vector<float> Utils::elementWiseSum(std::vector<float>& vector, float element);
template std::vector<int> Utils::greater(std::vector<float>& vector, float threshold);
template void Utils::deleteElementPositions(std::vector<float>& vector, std::vector<int>& positions);
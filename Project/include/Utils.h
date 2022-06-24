#ifndef UTILS_H
#define UTILS_H

//STL
#include <vector>
#include <algorithm>
#include <numeric>
#include <exception>

/**
* This class is used to contain some of the utilities that in c++ are not present w.r.t python (i.e. numpy library python)
* @author : Francesco Caldivezzi
*/
class Utils
{
public:

	/**
	* This function returns the indexes of the corresponding sorted vector as positions of the orginal vector i.e. a = [2,4,3] b = [2,3,4] ret = [0,2,1]
	* @param vector : The vector to compute the sorted indices
	* @typename T : Template function
	* @return : Vector of the sorted indices
	*/
	template <typename T>
	static std::vector<int>  argSort(const std::vector<T>& vector);

	/**
	* This function is used to implement the slicing in C++
	* @param vector : The vector to slice
	* @param start : Starting index from which we will slice
	* @param end : Ending index from which we will slice  
	* @typename T : Template function
	* @return : The sliced vector
	*/
	template <typename T>
	static std::vector<T> slice(std::vector<T>& vector, int start, int end);

	/**
	* This function is used to implement the slicing in C++
	* @param vector : The vector to slice
	* @param indices : The indices to consider for the slicing (can be not continue and also not sorted)
	* @typename T : Template function
	* @return : The sliced vector
	*/
	template <typename T>
	static std::vector<T> slice(std::vector<T>& vector, std::vector<int>& indices);

	/**
	* This function compute the maximum between a vector of elements and an element
	* @param vector : The vector to compare
	* @param element : The element to compare
	* @return :	The maximum between vector and element
	*/
	template <typename T>
	static T maximum(std::vector<T>& vector, T& element);


	/**
	* This function compute the element wise product between two vectors
	* @param vector1 : 1st vector
	* @param vector2 : 2nd vector
	* @return : Element wise product vector
	*/
	template <typename T>
	static std::vector<T> elementWiseProduct(std::vector<T>& vector1, std::vector<T>& vector2);

	/**
	* This function compute the element wise difference between two vectors
	* @param vector1 : 1st vector
	* @param vector2 : 2nd vector
	* @return : Element wise difference vector
	*/
	template <typename T>
	static std::vector<T> elementWiseDifference(std::vector<T>& vector1, std::vector<T>& vector2);

	/**
	* This function compute the element wise sum between one vector and one element
	* @param vector1 : Vector
	* @param element : Elemet to some to each component of the vector
	* @return : Element wise sum vector
	*/
	template <typename T>
	static std::vector<T> elementWiseSum(std::vector<T>& vector, T& element);


};

#endif // !UTILS_H


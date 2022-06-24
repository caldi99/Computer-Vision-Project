#ifndef UTILS_H
#define UTILS_H

//STL
#include <vector>
#include <algorithm>
#include <numeric>
#include <exception>

//OPENCV
#include <opencv2/highgui.hpp>

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
	* This function compute the element wise maximum between a vector of elements and an element
	* @param vector : The vector to compare
	* @param element : The element to compare
	* @typename T : Template function
	* @return :	The maximum vector of maximums
	*/
	template <typename T>
	static std::vector<T> elementWiseMaximum(std::vector<T>& vector, T element);


	/**
	* This function compute the element wise product between two vectors
	* @param vector1 : 1st vector
	* @param vector2 : 2nd vector
	* @typename T : Template function
	* @return : Element wise product vector
	*/
	template <typename T>
	static std::vector<T> elementWiseProduct(std::vector<T>& vector1, std::vector<T>& vector2);

	/**
	* This function compute the element wise difference between two vectors
	* @param vector1 : 1st vector
	* @param vector2 : 2nd vector
	* @typename T : Template function
	* @return : Element wise difference vector
	*/
	template <typename T>
	static std::vector<T> elementWiseDifference(std::vector<T>& vector1, std::vector<T>& vector2);

	/**
	* This function compute the element wise divion between two vectors
	* @param vector1 : 1st vector
	* @param vector2 : 2nd vector
	* @typename T : Template function
	* @return : Element wise division vector
	*/
	template <typename T>
	static std::vector<T> elementWiseDivision(std::vector<T>& vector1, std::vector<T>& vector2);

	/**
	* This function compute the element wise sum between one vector and one element
	* @param vector1 : Vector
	* @param element : Elemet to some to each component of the vector
	* @typename T : Template function
	* @return : Element wise sum vector
	*/
	template <typename T>
	static std::vector<T> elementWiseSum(std::vector<T>& vector, T element);

	/**
	* This function return the vector of all the posistions of the elements greater than the threshold specified
	* @param vector :
	* @param threshold :
	* @typename T : Template function
	* @return : The positions of the elements of the vector greater than the threshold
	*/
	template<typename T>
	static std::vector<int> greater(std::vector<T>& vector, T threshold);

	/**
	* This function remove from the vector provided all the elements in the positions specified by positions
	* @param vector : The vector for which removing the elements
	* @param positions : The positions of the elements that must be removed
	* @typename T : Template function
	*/
	template<typename T>
	static void deleteElementPositions(std::vector<T>& vector, std::vector<int>& positions);
};

#endif // !UTILS_H


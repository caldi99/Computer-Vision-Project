#ifndef UTILS_H
#define UTILS_H

//STL
#include <vector>
#include <algorithm>
#include <numeric>

/**
* This class is used to contain some of the utilities that in c++ are not present w.r.t python
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
	static std::vector<T> slicing(std::vector<T>& vector, int start, int end);
};

#endif // !UTILS_H


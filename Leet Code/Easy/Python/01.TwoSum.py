""" Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.
 You may assume that each input would have exactly one solution, and you may not use the same element twice.
 You can return the answer in any order. 
Example 1:

Input: nums = [2,7,11,15], target = 9
Output: [0,1]
Explanation: Because nums[0] + nums[1] == 9, we return [0, 1].
Example 2:

Input: nums = [3,2,4], target = 6
Output: [1,2] 

"""

class Solution(object):
    def __init__(self):
        self.storage={}
    def addResult(self, number1, number2):
        return number1+number2
    def twoSum(self, nums, target):        
        for i in range(len(nums)):
            remainder = target - nums[i]
            if(remainder in self.storage.keys()):
                return [self.storage[remainder], i]
            else:
                self.storage[nums[i]]=i 

obj = Solution()
print(obj.twoSum([-3,4,3,90],0))
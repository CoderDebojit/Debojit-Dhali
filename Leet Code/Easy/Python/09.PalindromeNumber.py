"""
Given an integer x, return true if x is palindrome integer.
An integer is a palindrome when it reads the same backward as forward.
For example, 121 is a palindrome while 123 is not.

Example 1:

Input: x = 121
Output: true
Explanation: 121 reads as 121 from left to right and from right to left.
Example 2:

Input: x = -121
Output: false
Explanation: From left to right, it reads -121. From right to left, it becomes 121-. Therefore it is not a palindrome.
"""

class Solution(object):
    def isPalindrome(self, x):
        """
        :type x: int
        :rtype: bool
        """
        if(x >= 0):
            if(x>=9):
                temp=x
                rev = 0
                while temp > 0:
                    lastDegit = temp % 10
                    rev = rev*10 + lastDegit

                    temp=temp//10
                if(x==rev):
                    return True
                return False
            return True
        return False


print(Solution().isPalindrome(101))
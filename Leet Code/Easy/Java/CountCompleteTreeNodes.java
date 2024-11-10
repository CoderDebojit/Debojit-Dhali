/*
Given the root of a complete binary tree, return the number of the nodes in the tree.

According to Wikipedia, every level, except possibly the last, is completely filled in a complete binary tree, and all nodes in the last level are as far left as possible. It can have between 1 and 2h nodes inclusive at the last level h.

Design an algorithm that runs in less than O(n) time complexity.
*/

class Demo {
    public static void main(String args[]) // static method
    {
        CountCompleteTreeNodes root = new CountCompleteTreeNodes(2, new CountCompleteTreeNodes(4), new CountCompleteTreeNodes(5));

        int result = new Solution().countNodes(root);

        System.out.println(result);
    }
}
//Definition for a binary tree node.
public class CountCompleteTreeNodes {
    int val;
    CountCompleteTreeNodes left;
    CountCompleteTreeNodes right;

    CountCompleteTreeNodes() {
    }

    CountCompleteTreeNodes(int val) {
        this.val = val;
    }

    CountCompleteTreeNodes(int val, CountCompleteTreeNodes left, CountCompleteTreeNodes right) {
        this.val = val;
        this.left = left;
        this.right = right;
    }
}

class Solution {
    public int countNodes(CountCompleteTreeNodes root) {
        if (root == null) {
            return 0;
        }

        return 1 + countNodes(root.left) + countNodes(root.right);
    }
}


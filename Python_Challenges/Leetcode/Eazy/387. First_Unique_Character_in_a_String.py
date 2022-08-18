"""
Problem - https://leetcode.com/problems/first-unique-character-in-a-string/


Eazy
Given a string, find the first non-repeating character in it and return it's index. If it doesn't exist, return -1.

Examples:

s = "leetcode"
return 0.

s = "loveleetcode",
return 2.
Note: You may assume the string contain only lowercase letters.

"""

# 1st Alternative
def firstUniqChar(s: str) -> int:
    d = dict()
    for c in s:
        d[c] = d.get(c, 0) + 1
    print(d)
    """
    {'l': 1, 'e': 3, 't': 1, 'c': 1, 'o': 1, 'd': 1}
    """

    for i in range(len(s)):
        if d[s[i]] == 1:
            return i

    return -1


# 2nd Alternative
def firstUniqChar(s: str) -> int:
    chars = set(s)
    print(chars)
    return min([s.index(char) for char in chars if s.count(char) == 1] or [-1])


# 3rd Alternative


def firstUniqChar(s: str) -> int:
    return min(
        (s.index(l) for l in "abcdefghijklmnopqrstuvwxyz" if s.count(l) == 1),
        default=-1,
    )


s = "leetcode"
print(firstUniqChar(s))


s = "loveleetcode"
print(firstUniqChar(s))

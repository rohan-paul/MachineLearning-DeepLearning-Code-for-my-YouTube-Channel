# https://www.codewars.com/kata/51689e27fe9a00b126000004

"""

Complete the method so that it formats the words into a single comma separated value. The last word should be separated by the word 'and' instead of a comma. The method takes in an array of strings and returns a single formatted string. Empty string values should be ignored. Empty arrays or null/nil values being passed into the method should result in an empty string being returned.

format_words(['ninja', 'samurai', 'ronin']) # should return "ninja, samurai and ronin"

format_words(['ninja', '', 'ronin']) # should return "ninja and ronin"

format_words([]) # should return ""

"""


def format_words(words):
    result = ""
    if words:
        # The filter() function extracts elements from an iterable (list, tuple etc.) for which a function returns True
        # So it takes in a function and a list as arguments.
        words = list(filter(lambda x: x != "", words))
        if len(words) <= 2:
            result += " and ".join(words)
        else:
            result += ", ".join(words[:-2]) + ", " + " and ".join(words[-2:])
    return result


print(format_words(["one", "two", "three", "four"]))
# format_words(["one", "two", "three", "four"])

a = ["one", "two", "three", "four"]
# print(a[:-2]) # ['one', 'two']
# print(a[-2:])
# print(" and ".join(a))
# print(", ".join(a for a in a if a)[::-1])
print(", ".join(a for a in a if a)[::-1].replace(",", "dna ", 1)[::-1])


# ******* SOME OTHER REFACTORS *********


def format_words(words):
    return (
        ", ".join(word for word in words if word)[::-1].replace(",", "dna ", 1)[::-1]
        if words
        else ""
    )


# print(format_words(a))

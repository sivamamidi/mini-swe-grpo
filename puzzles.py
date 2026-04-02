"""
Coding Puzzles — The "bugs" our RL agent will learn to fix.

Each puzzle is a dict with:
  - id:          unique name
  - description: what the function SHOULD do (the "issue")
  - buggy_code:  the broken implementation (the "codebase")
  - tests:       list of assert statements that must ALL pass
  - difficulty:  1 (easy) to 3 (hard) — used for curriculum training later

These are intentionally simple so we can train on a Mac / single GPU.
Real SWE-bench has 2,294 problems from Django, scikit-learn, sympy, etc.
Our mini version captures the same idea: read the bug, write the fix, pass the tests.
"""

PUZZLES = [
    # ── Difficulty 1: One-line fixes ──────────────────────────────────
    {
        "id": "fix_add",
        "description": "The function `add(a, b)` should return the sum of a and b, but it returns the wrong result.",
        "buggy_code": "def add(a, b):\n    return a - b",
        "tests": [
            "assert add(2, 3) == 5",
            "assert add(-1, 1) == 0",
            "assert add(0, 0) == 0",
        ],
        "difficulty": 1,
    },
    {
        "id": "fix_max",
        "description": "The function `find_max(lst)` should return the largest element in a list, but it returns the smallest.",
        "buggy_code": "def find_max(lst):\n    return min(lst)",
        "tests": [
            "assert find_max([1, 2, 3]) == 3",
            "assert find_max([-5, -1, -3]) == -1",
            "assert find_max([42]) == 42",
        ],
        "difficulty": 1,
    },
    {
        "id": "fix_is_even",
        "description": "The function `is_even(n)` should return True if n is even, False otherwise. It currently has the logic inverted.",
        "buggy_code": "def is_even(n):\n    return n % 2 != 0",
        "tests": [
            "assert is_even(4) == True",
            "assert is_even(7) == False",
            "assert is_even(0) == True",
        ],
        "difficulty": 1,
    },
    {
        "id": "fix_abs",
        "description": "The function `absolute(n)` should return the absolute value of n. It currently returns n unchanged.",
        "buggy_code": "def absolute(n):\n    return n",
        "tests": [
            "assert absolute(-5) == 5",
            "assert absolute(3) == 3",
            "assert absolute(0) == 0",
        ],
        "difficulty": 1,
    },
    {
        "id": "fix_string_length",
        "description": "The function `string_length(s)` should return the number of characters in the string. It returns the string itself.",
        "buggy_code": "def string_length(s):\n    return s",
        "tests": [
            "assert string_length('hello') == 5",
            "assert string_length('') == 0",
            "assert string_length('ab') == 2",
        ],
        "difficulty": 1,
    },

    # ── Difficulty 2: Requires understanding the logic ────────────────
    {
        "id": "fix_factorial",
        "description": "The function `factorial(n)` should return n! (n factorial). It has an off-by-one error in the range.",
        "buggy_code": "def factorial(n):\n    result = 1\n    for i in range(1, n):  # bug: should be n+1\n        result *= i\n    return result",
        "tests": [
            "assert factorial(5) == 120",
            "assert factorial(1) == 1",
            "assert factorial(0) == 1",
        ],
        "difficulty": 2,
    },
    {
        "id": "fix_reverse_string",
        "description": "The function `reverse_string(s)` should return the string reversed. It currently returns the string sorted alphabetically.",
        "buggy_code": "def reverse_string(s):\n    return ''.join(sorted(s))",
        "tests": [
            "assert reverse_string('hello') == 'olleh'",
            "assert reverse_string('ab') == 'ba'",
            "assert reverse_string('a') == 'a'",
        ],
        "difficulty": 2,
    },
    {
        "id": "fix_count_vowels",
        "description": "The function `count_vowels(s)` should count the number of vowels (a,e,i,o,u) in a string. It currently counts consonants instead.",
        "buggy_code": "def count_vowels(s):\n    count = 0\n    for char in s:\n        if char.lower() not in 'aeiou':\n            count += 1\n    return count",
        "tests": [
            "assert count_vowels('hello') == 2",
            "assert count_vowels('aeiou') == 5",
            "assert count_vowels('xyz') == 0",
        ],
        "difficulty": 2,
    },
    {
        "id": "fix_fibonacci",
        "description": "The function `fibonacci(n)` should return the nth Fibonacci number (0-indexed: fib(0)=0, fib(1)=1, fib(2)=1, ...). The base cases are swapped.",
        "buggy_code": "def fibonacci(n):\n    if n == 0:\n        return 1  # bug: should be 0\n    if n == 1:\n        return 0  # bug: should be 1\n    return fibonacci(n-1) + fibonacci(n-2)",
        "tests": [
            "assert fibonacci(0) == 0",
            "assert fibonacci(1) == 1",
            "assert fibonacci(6) == 8",
        ],
        "difficulty": 2,
    },
    {
        "id": "fix_is_palindrome",
        "description": "The function `is_palindrome(s)` should return True if the string reads the same forwards and backwards. The comparison is wrong.",
        "buggy_code": "def is_palindrome(s):\n    return s == s[1:]",
        "tests": [
            "assert is_palindrome('racecar') == True",
            "assert is_palindrome('hello') == False",
            "assert is_palindrome('aba') == True",
        ],
        "difficulty": 2,
    },

    # ── Difficulty 3: Multi-line fixes / algorithmic ──────────────────
    {
        "id": "fix_bubble_sort",
        "description": "The function `bubble_sort(lst)` should sort a list in ascending order using bubble sort. The swap condition is inverted, so it sorts descending.",
        "buggy_code": "def bubble_sort(lst):\n    lst = lst.copy()\n    n = len(lst)\n    for i in range(n):\n        for j in range(0, n-i-1):\n            if lst[j] < lst[j+1]:  # bug: should be >\n                lst[j], lst[j+1] = lst[j+1], lst[j]\n    return lst",
        "tests": [
            "assert bubble_sort([3, 1, 2]) == [1, 2, 3]",
            "assert bubble_sort([5, 4, 3, 2, 1]) == [1, 2, 3, 4, 5]",
            "assert bubble_sort([1]) == [1]",
        ],
        "difficulty": 3,
    },
    {
        "id": "fix_flatten_list",
        "description": "The function `flatten(lst)` should flatten a nested list. Example: flatten([1, [2, [3]]]) -> [1, 2, 3]. It currently only flattens one level deep.",
        "buggy_code": "def flatten(lst):\n    result = []\n    for item in lst:\n        if isinstance(item, list):\n            result.extend(item)  # bug: should recurse\n        else:\n            result.append(item)\n    return result",
        "tests": [
            "assert flatten([1, [2, [3]]]) == [1, 2, 3]",
            "assert flatten([[1, 2], [3, 4]]) == [1, 2, 3, 4]",
            "assert flatten([1, 2, 3]) == [1, 2, 3]",
        ],
        "difficulty": 3,
    },
    {
        "id": "fix_binary_search",
        "description": "The function `binary_search(lst, target)` should return the index of target in a sorted list, or -1 if not found. The mid-point calculation and boundary updates have bugs.",
        "buggy_code": "def binary_search(lst, target):\n    low, high = 0, len(lst)\n    while low < high:\n        mid = (low + high) // 2\n        if lst[mid] == target:\n            return mid\n        elif lst[mid] < target:\n            low = mid  # bug: should be mid + 1\n        else:\n            high = mid\n    return -1",
        "tests": [
            "assert binary_search([1, 3, 5, 7, 9], 5) == 2",
            "assert binary_search([1, 3, 5, 7, 9], 1) == 0",
            "assert binary_search([1, 3, 5, 7, 9], 6) == -1",
        ],
        "difficulty": 3,
    },
    {
        "id": "fix_matrix_transpose",
        "description": "The function `transpose(matrix)` should return the transpose of a 2D matrix. Rows become columns. The indexing is wrong.",
        "buggy_code": "def transpose(matrix):\n    rows = len(matrix)\n    cols = len(matrix[0])\n    result = [[0] * rows for _ in range(cols)]\n    for i in range(rows):\n        for j in range(cols):\n            result[j][i] = matrix[i][i]  # bug: should be matrix[i][j]\n    return result",
        "tests": [
            "assert transpose([[1, 2], [3, 4]]) == [[1, 3], [2, 4]]",
            "assert transpose([[1, 2, 3]]) == [[1], [2], [3]]",
            "assert transpose([[1], [2], [3]]) == [[1, 2, 3]]",
        ],
        "difficulty": 3,
    },
    {
        "id": "fix_remove_duplicates",
        "description": "The function `remove_duplicates(lst)` should return a new list with duplicates removed, preserving order. It currently removes ALL occurrences instead of keeping the first.",
        "buggy_code": "def remove_duplicates(lst):\n    result = []\n    for item in lst:\n        if lst.count(item) == 1:  # bug: should check if item already in result\n            result.append(item)\n    return result",
        "tests": [
            "assert remove_duplicates([1, 2, 2, 3, 1]) == [1, 2, 3]",
            "assert remove_duplicates([1, 1, 1]) == [1]",
            "assert remove_duplicates([1, 2, 3]) == [1, 2, 3]",
        ],
        "difficulty": 3,
    },
]

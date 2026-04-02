"""
Hard Coding Puzzles — Problems that a 1.5B model will struggle with.

These puzzles are designed to be tricky:
  - Less obvious bug descriptions (no "the bug is X")
  - Multiple interacting bugs
  - Requires algorithmic reasoning
  - Edge cases that trip up small models

The goal: create a puzzle set where the base model solves ~30-50%.
That gives RL room to improve.
"""

PUZZLES_HARD = [
    # ── Difficulty 1: Subtle single-line bugs ─────────────────────────
    {
        "id": "hard_off_by_one",
        "description": "The function `sum_range(a, b)` should return the sum of all integers from a to b, inclusive. It returns the wrong value for some inputs.",
        "buggy_code": "def sum_range(a, b):\n    total = 0\n    for i in range(a, b):\n        total += i\n    return total",
        "tests": [
            "assert sum_range(1, 5) == 15",
            "assert sum_range(3, 3) == 3",
            "assert sum_range(0, 0) == 0",
            "assert sum_range(-2, 2) == 0",
        ],
        "difficulty": 1,
    },
    {
        "id": "hard_default_mutable",
        "description": "The function `append_to(item, lst=[])` should append item to lst and return lst. However, calling it multiple times without providing lst gives unexpected results. Fix the mutable default argument bug.",
        "buggy_code": "def append_to(item, lst=[]):\n    lst.append(item)\n    return lst",
        "tests": [
            "assert append_to(1) == [1]",
            "assert append_to(2) == [2]",
            "assert append_to(3, [10]) == [10, 3]",
        ],
        "difficulty": 1,
    },
    {
        "id": "hard_integer_division",
        "description": "The function `average(numbers)` should return the arithmetic mean as a float. It returns an integer instead of a float for some inputs.",
        "buggy_code": "def average(numbers):\n    return sum(numbers) // len(numbers)",
        "tests": [
            "assert average([1, 2, 3, 4]) == 2.5",
            "assert average([10]) == 10.0",
            "assert average([1, 2]) == 1.5",
        ],
        "difficulty": 1,
    },
    {
        "id": "hard_scope_bug",
        "description": "The function `make_counters(n)` should return a list of n functions, where the i-th function returns i when called. Currently all returned functions return the same value.",
        "buggy_code": "def make_counters(n):\n    counters = []\n    for i in range(n):\n        counters.append(lambda: i)\n    return counters",
        "tests": [
            "assert make_counters(3)[0]() == 0",
            "assert make_counters(3)[1]() == 1",
            "assert make_counters(3)[2]() == 2",
            "assert make_counters(1)[0]() == 0",
        ],
        "difficulty": 1,
    },
    {
        "id": "hard_string_compare",
        "description": "The function `is_anagram(s1, s2)` should return True if s1 and s2 are anagrams (same letters, different order), case-insensitive. It fails on mixed-case inputs.",
        "buggy_code": "def is_anagram(s1, s2):\n    return sorted(s1) == sorted(s2)",
        "tests": [
            "assert is_anagram('Listen', 'Silent') == True",
            "assert is_anagram('hello', 'world') == False",
            "assert is_anagram('Astronomer', 'Moon starer') == False",
            "assert is_anagram('abc', 'cba') == True",
        ],
        "difficulty": 1,
    },

    # ── Difficulty 2: Multi-bug / algorithmic ─────────────────────────
    {
        "id": "hard_deep_copy",
        "description": "The function `duplicate_grid(grid)` should return a deep copy of a 2D grid. Modifying the copy should NOT affect the original. Currently modifications to the copy affect the original.",
        "buggy_code": "def duplicate_grid(grid):\n    return [row for row in grid]",
        "tests": [
            "g = [[1,2],[3,4]]; c = duplicate_grid(g); c[0][0] = 99; assert g[0][0] == 1",
            "g = [[1]]; c = duplicate_grid(g); assert c == [[1]]",
            "g = [[1,2],[3,4]]; c = duplicate_grid(g); c[1][1] = 0; assert g[1][1] == 4",
        ],
        "difficulty": 2,
    },
    {
        "id": "hard_spiral_order",
        "description": "The function `spiral_order(matrix)` should return all elements of the matrix in spiral order (right, down, left, up, repeat). It produces wrong output for non-square matrices.",
        "buggy_code": "def spiral_order(matrix):\n    result = []\n    if not matrix:\n        return result\n    top, bottom, left, right = 0, len(matrix) - 1, 0, len(matrix[0]) - 1\n    while top <= bottom and left <= right:\n        for i in range(left, right + 1):\n            result.append(matrix[top][i])\n        top += 1\n        for i in range(top, bottom + 1):\n            result.append(matrix[i][right])\n        right -= 1\n        for i in range(right, left - 1, -1):\n            result.append(matrix[bottom][i])\n        bottom -= 1\n        for i in range(bottom, top - 1, -1):\n            result.append(matrix[i][left])\n        left += 1\n    return result",
        "tests": [
            "assert spiral_order([[1,2,3],[4,5,6],[7,8,9]]) == [1,2,3,6,9,8,7,4,5]",
            "assert spiral_order([[1,2,3,4]]) == [1,2,3,4]",
            "assert spiral_order([[1],[2],[3]]) == [1,2,3]",
        ],
        "difficulty": 2,
    },
    {
        "id": "hard_lru_cache",
        "description": "The class `LRUCache` should implement a Least Recently Used cache with a given capacity. `get(key)` returns the value or -1. `put(key, value)` inserts/updates and evicts the least recently used item if over capacity. The eviction logic is broken.",
        "buggy_code": "class LRUCache:\n    def __init__(self, capacity):\n        self.capacity = capacity\n        self.cache = {}\n        self.order = []\n\n    def get(self, key):\n        if key in self.cache:\n            return self.cache[key]\n        return -1\n\n    def put(self, key, value):\n        if key in self.cache:\n            self.cache[key] = value\n        else:\n            if len(self.cache) >= self.capacity:\n                del self.cache[self.order[0]]\n                self.order.pop(0)\n            self.cache[key] = value\n            self.order.append(key)",
        "tests": [
            "c = LRUCache(2); c.put(1, 1); c.put(2, 2); assert c.get(1) == 1; c.put(3, 3); assert c.get(2) == -1",
            "c = LRUCache(1); c.put(1, 1); c.put(2, 2); assert c.get(1) == -1; assert c.get(2) == 2",
            "c = LRUCache(2); c.put(1, 1); c.put(2, 2); c.get(1); c.put(3, 3); assert c.get(1) == 1; assert c.get(2) == -1",
        ],
        "difficulty": 2,
    },
    {
        "id": "hard_merge_intervals",
        "description": "The function `merge_intervals(intervals)` takes a list of [start, end] pairs and merges overlapping intervals. It fails when intervals are not pre-sorted.",
        "buggy_code": "def merge_intervals(intervals):\n    if not intervals:\n        return []\n    merged = [intervals[0]]\n    for start, end in intervals[1:]:\n        if start <= merged[-1][1]:\n            merged[-1][1] = max(merged[-1][1], end)\n        else:\n            merged.append([start, end])\n    return merged",
        "tests": [
            "assert merge_intervals([[1,3],[2,6],[8,10],[15,18]]) == [[1,6],[8,10],[15,18]]",
            "assert merge_intervals([[1,4],[4,5]]) == [[1,5]]",
            "assert merge_intervals([[5,8],[1,3],[2,4]]) == [[1,4],[5,8]]",
            "assert merge_intervals([]) == []",
        ],
        "difficulty": 2,
    },
    {
        "id": "hard_balanced_parens",
        "description": "The function `is_balanced(s)` should check if a string of brackets ()[]{}  is balanced. It only checks parentheses and ignores [] and {}.",
        "buggy_code": "def is_balanced(s):\n    stack = []\n    for char in s:\n        if char == '(':\n            stack.append(char)\n        elif char == ')':\n            if not stack:\n                return False\n            stack.pop()\n    return len(stack) == 0",
        "tests": [
            "assert is_balanced('()[]{}') == True",
            "assert is_balanced('([{}])') == True",
            "assert is_balanced('(]') == False",
            "assert is_balanced('{[}]') == False",
            "assert is_balanced('') == True",
        ],
        "difficulty": 2,
    },

    # ── Difficulty 3: Algorithm design / tricky edge cases ────────────
    {
        "id": "hard_longest_subseq",
        "description": "The function `longest_increasing_subsequence(nums)` should return the LENGTH of the longest strictly increasing subsequence. It returns incorrect results.",
        "buggy_code": "def longest_increasing_subsequence(nums):\n    if not nums:\n        return 0\n    dp = [1] * len(nums)\n    for i in range(1, len(nums)):\n        for j in range(i):\n            if nums[j] < nums[i]:\n                dp[i] = dp[j] + 1  # bug: should be max(dp[i], dp[j] + 1)\n    return max(dp)",
        "tests": [
            "assert longest_increasing_subsequence([10, 9, 2, 5, 3, 7, 101, 18]) == 4",
            "assert longest_increasing_subsequence([0, 1, 0, 3, 2, 3]) == 4",
            "assert longest_increasing_subsequence([7, 7, 7, 7]) == 1",
            "assert longest_increasing_subsequence([]) == 0",
        ],
        "difficulty": 3,
    },
    {
        "id": "hard_knapsack",
        "description": "The function `knapsack(weights, values, capacity)` should return the maximum value achievable with items that fit in the given capacity (0/1 knapsack). The DP table initialization or transition is wrong.",
        "buggy_code": "def knapsack(weights, values, capacity):\n    n = len(weights)\n    dp = [[0] * (capacity + 1) for _ in range(n + 1)]\n    for i in range(1, n + 1):\n        for w in range(1, capacity + 1):\n            if weights[i-1] <= w:\n                dp[i][w] = values[i-1] + dp[i-1][w - weights[i-1]]\n            else:\n                dp[i][w] = dp[i-1][w]\n    return dp[n][capacity]",
        "tests": [
            "assert knapsack([1, 2, 3], [6, 10, 12], 5) == 22",
            "assert knapsack([2, 3, 4, 5], [3, 4, 5, 6], 5) == 7",
            "assert knapsack([10], [100], 5) == 0",
            "assert knapsack([], [], 10) == 0",
        ],
        "difficulty": 3,
    },
    {
        "id": "hard_trie_insert_search",
        "description": "The class `Trie` should support `insert(word)` and `search(word)` (exact match) and `starts_with(prefix)`. The search method incorrectly returns True for prefixes that aren't complete words.",
        "buggy_code": "class Trie:\n    def __init__(self):\n        self.children = {}\n        self.is_end = False\n\n    def insert(self, word):\n        node = self\n        for ch in word:\n            if ch not in node.children:\n                node.children[ch] = Trie()\n            node = node.children[ch]\n        node.is_end = True\n\n    def search(self, word):\n        node = self\n        for ch in word:\n            if ch not in node.children:\n                return False\n            node = node.children[ch]\n        return True  # bug: should check node.is_end\n\n    def starts_with(self, prefix):\n        node = self\n        for ch in prefix:\n            if ch not in node.children:\n                return False\n            node = node.children[ch]\n        return True",
        "tests": [
            "t = Trie(); t.insert('apple'); assert t.search('apple') == True",
            "t = Trie(); t.insert('apple'); assert t.search('app') == False",
            "t = Trie(); t.insert('apple'); assert t.starts_with('app') == True",
            "t = Trie(); t.insert('apple'); t.insert('app'); assert t.search('app') == True",
        ],
        "difficulty": 3,
    },
    {
        "id": "hard_graph_cycle",
        "description": "The function `has_cycle(graph)` takes an adjacency list (dict of node -> list of neighbors) for a DIRECTED graph and should return True if the graph contains a cycle. It incorrectly detects cycles in DAGs.",
        "buggy_code": "def has_cycle(graph):\n    visited = set()\n    \n    def dfs(node):\n        if node in visited:\n            return True\n        visited.add(node)\n        for neighbor in graph.get(node, []):\n            if dfs(neighbor):\n                return True\n        return False\n    \n    for node in graph:\n        if dfs(node):\n            return True\n    return False",
        "tests": [
            "assert has_cycle({0: [1], 1: [2], 2: [0]}) == True",
            "assert has_cycle({0: [1], 1: [2], 2: []}) == False",
            "assert has_cycle({0: [1, 2], 1: [2], 2: []}) == False",
            "assert has_cycle({}) == False",
            "assert has_cycle({0: [1], 1: [2], 2: [1]}) == True",
        ],
        "difficulty": 3,
    },
    {
        "id": "hard_eval_rpn",
        "description": "The function `eval_rpn(tokens)` evaluates a Reverse Polish Notation expression. Tokens are strings: numbers or operators (+, -, *, /). Division should truncate toward zero. It mishandles negative division.",
        "buggy_code": "def eval_rpn(tokens):\n    stack = []\n    for token in tokens:\n        if token in '+-*/':\n            b = stack.pop()\n            a = stack.pop()\n            if token == '+':\n                stack.append(a + b)\n            elif token == '-':\n                stack.append(a - b)\n            elif token == '*':\n                stack.append(a * b)\n            elif token == '/':\n                stack.append(a // b)\n        else:\n            stack.append(int(token))\n    return stack[0]",
        "tests": [
            "assert eval_rpn(['2', '1', '+', '3', '*']) == 9",
            "assert eval_rpn(['4', '13', '5', '/', '+']) == 6",
            "assert eval_rpn(['10', '6', '9', '3', '+', '-11', '*', '/', '*', '17', '+', '5', '+']) == 22",
        ],
        "difficulty": 3,
    },
]

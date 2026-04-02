"""
Medium Coding Puzzles — Targeting the "sweet spot" for GRPO training.

These puzzles are designed so a 1.5B code model (Qwen2.5-Coder-1.5B) solves
them ~30-60% of the time at temperature=0.8. They exploit:
  - Python-specific semantics (shallow copy, generator exhaustion, float quirks)
  - Off-by-one errors in non-obvious contexts
  - Bugs requiring understanding of TWO interacting pieces of logic
  - Recursive functions with subtle base case errors
  - Class methods with state mutation bugs

Descriptions deliberately do NOT reveal the exact bug — the model must diagnose it.
"""

PUZZLES_MEDIUM = [
    # ── 1. Shallow copy trap in nested accumulation ───────────────────
    {
        "id": "med_shallow_copy_accumulate",
        "description": "The function `group_by_parity(nums)` should return a dict with keys 'even' and 'odd', each mapping to a list of numbers. When called multiple times, results from previous calls should not leak into later ones.",
        "buggy_code": "def group_by_parity(nums, result={'even': [], 'odd': []}):\n    for n in nums:\n        if n % 2 == 0:\n            result['even'].append(n)\n        else:\n            result['odd'].append(n)\n    return result",
        "tests": [
            "assert group_by_parity([1, 2, 3]) == {'even': [2], 'odd': [1, 3]}",
            "assert group_by_parity([4, 5]) == {'even': [4], 'odd': [5]}",
            "assert group_by_parity([]) == {'even': [], 'odd': []}",
            "assert group_by_parity([2, 2]) == {'even': [2, 2], 'odd': []}",
        ],
        "difficulty": 2,
    },

    # ── 2. Generator exhaustion after first pass ──────────────────────
    {
        "id": "med_generator_exhaustion",
        "description": "The function `sum_and_max(numbers)` takes an iterable of numbers and should return a tuple (total_sum, maximum). It works with lists but returns wrong results when given a generator.",
        "buggy_code": "def sum_and_max(numbers):\n    total = sum(numbers)\n    maximum = max(numbers)\n    return (total, maximum)",
        "tests": [
            "assert sum_and_max(x for x in [1, 2, 3]) == (6, 3)",
            "assert sum_and_max(x for x in [10]) == (10, 10)",
            "assert sum_and_max([4, 5, 6]) == (15, 6)",
            "assert sum_and_max(x for x in [-1, -2, -3]) == (-6, -1)",
        ],
        "difficulty": 2,
    },

    # ── 3. Float precision comparison ─────────────────────────────────
    {
        "id": "med_float_equality",
        "description": "The function `remove_value(lst, target)` should return a new list with all elements equal to target removed. It works for integers but fails for certain float values that should be considered equal.",
        "buggy_code": "def remove_value(lst, target):\n    return [x for x in lst if x != target]",
        "tests": [
            "assert remove_value([0.1 + 0.2, 0.5, 0.7], 0.3) == [0.5, 0.7]",
            "assert remove_value([1.0, 2.0, 3.0], 2.0) == [1.0, 3.0]",
            "assert remove_value([0.1 + 0.1 + 0.1, 0.4], 0.3) == [0.4]",
            "assert remove_value([], 1.0) == []",
        ],
        "difficulty": 2,
    },

    # ── 4. Recursive power with wrong base case ──────────────────────
    {
        "id": "med_recursive_power",
        "description": "The function `power(base, exp)` should compute base raised to exp (non-negative integer) using recursion. It returns incorrect results for certain inputs.",
        "buggy_code": "def power(base, exp):\n    if exp == 1:\n        return base\n    return base * power(base, exp - 1)",
        "tests": [
            "assert power(2, 0) == 1",
            "assert power(2, 3) == 8",
            "assert power(5, 1) == 5",
            "assert power(3, 4) == 81",
            "assert power(10, 0) == 1",
        ],
        "difficulty": 2,
    },

    # ── 5. String deduplication with case folding interaction ─────────
    {
        "id": "med_case_dedup",
        "description": "The function `unique_words(sentence)` should return a list of unique words from the sentence, case-insensitive, preserving the case of the FIRST occurrence and the order of first appearance.",
        "buggy_code": "def unique_words(sentence):\n    seen = set()\n    result = []\n    for word in sentence.split():\n        if word not in seen:\n            seen.add(word)\n            result.append(word)\n    return result",
        "tests": [
            "assert unique_words('The the THE') == ['The']",
            "assert unique_words('Hello World hello') == ['Hello', 'World']",
            "assert unique_words('a A b B a') == ['a', 'b']",
            "assert unique_words('one') == ['one']",
            "assert unique_words('') == []",
        ],
        "difficulty": 2,
    },

    # ── 6. Class with shared mutable class attribute ──────────────────
    {
        "id": "med_class_shared_state",
        "description": "The `Inventory` class tracks items. Each Inventory instance should have its own independent list of items. Creating multiple instances and adding items to one should not affect the other.",
        "buggy_code": "class Inventory:\n    items = []\n\n    def __init__(self, name):\n        self.name = name\n\n    def add(self, item):\n        self.items.append(item)\n        return self\n\n    def get_items(self):\n        return list(self.items)",
        "tests": [
            "a = Inventory('A'); b = Inventory('B'); a.add('x'); assert b.get_items() == []",
            "a = Inventory('A'); a.add('x'); a.add('y'); assert a.get_items() == ['x', 'y']",
            "a = Inventory('A'); b = Inventory('B'); a.add('x'); b.add('y'); assert a.get_items() == ['x']",
            "a = Inventory('A'); assert a.get_items() == []",
        ],
        "difficulty": 2,
    },

    # ── 7. Off-by-one in sliding window maximum ──────────────────────
    {
        "id": "med_sliding_window_avg",
        "description": "The function `moving_average(nums, k)` should return a list of averages of each contiguous window of size k. For nums=[1,2,3,4,5] and k=3, the result should be [2.0, 3.0, 4.0].",
        "buggy_code": "def moving_average(nums, k):\n    result = []\n    window_sum = sum(nums[:k])\n    result.append(window_sum / k)\n    for i in range(k, len(nums)):\n        window_sum += nums[i] - nums[i - k + 1]\n        result.append(window_sum / k)\n    return result",
        "tests": [
            "assert moving_average([1, 2, 3, 4, 5], 3) == [2.0, 3.0, 4.0]",
            "assert moving_average([10, 20, 30], 2) == [15.0, 25.0]",
            "assert moving_average([5], 1) == [5.0]",
            "assert moving_average([1, 1, 1, 1], 2) == [1.0, 1.0, 1.0]",
        ],
        "difficulty": 2,
    },

    # ── 8. Dict ordering + deletion during iteration ──────────────────
    {
        "id": "med_filter_dict",
        "description": "The function `filter_low_scores(scores, threshold)` takes a dict of {name: score} and should return a NEW dict containing only entries where score >= threshold. The original dict must not be modified.",
        "buggy_code": "def filter_low_scores(scores, threshold):\n    for name in scores:\n        if scores[name] < threshold:\n            del scores[name]\n    return scores",
        "tests": [
            "d = {'a': 90, 'b': 50, 'c': 70}; r = filter_low_scores(d, 60); assert r == {'a': 90, 'c': 70}",
            "d = {'a': 90, 'b': 50, 'c': 70}; filter_low_scores(d, 60); assert d == {'a': 90, 'b': 50, 'c': 70}",
            "assert filter_low_scores({'x': 10}, 5) == {'x': 10}",
            "assert filter_low_scores({'x': 10}, 20) == {}",
            "assert filter_low_scores({}, 50) == {}",
        ],
        "difficulty": 3,
    },

    # ── 9. Recursive flatten with depth limit ─────────────────────────
    {
        "id": "med_flatten_depth",
        "description": "The function `flatten(lst, depth)` should flatten a nested list up to `depth` levels. depth=1 flattens one level, depth=2 flattens two levels, etc. depth=0 means no flattening.",
        "buggy_code": "def flatten(lst, depth):\n    result = []\n    for item in lst:\n        if isinstance(item, list) and depth >= 0:\n            result.extend(flatten(item, depth - 1))\n        else:\n            result.append(item)\n    return result",
        "tests": [
            "assert flatten([1, [2, [3, [4]]]], 1) == [1, 2, [3, [4]]]",
            "assert flatten([1, [2, [3, [4]]]], 2) == [1, 2, 3, [4]]",
            "assert flatten([1, [2, [3]]], 0) == [1, [2, [3]]]",
            "assert flatten([[1, 2], [3, 4]], 1) == [1, 2, 3, 4]",
            "assert flatten([1, 2, 3], 5) == [1, 2, 3]",
        ],
        "difficulty": 2,
    },

    # ── 10. Two interacting bugs: accumulator + return ────────────────
    {
        "id": "med_running_total",
        "description": "The function `running_totals(nums)` should return a list where each element is the cumulative sum up to that index. For [1, 2, 3] it should return [1, 3, 6].",
        "buggy_code": "def running_totals(nums):\n    result = []\n    total = 0\n    for i in range(len(nums)):\n        total = total + nums[i]\n        result.append(total)\n    return result[:-1]",
        "tests": [
            "assert running_totals([1, 2, 3]) == [1, 3, 6]",
            "assert running_totals([5]) == [5]",
            "assert running_totals([1, 1, 1, 1]) == [1, 2, 3, 4]",
            "assert running_totals([]) == []",
        ],
        "difficulty": 2,
    },

    # ── 11. Recursive merge sort with wrong merge logic ──────────────
    {
        "id": "med_merge_sorted",
        "description": "The function `merge_sorted(a, b)` takes two SORTED lists and should return a single sorted list containing all elements from both. It produces incorrect results when one list is exhausted before the other.",
        "buggy_code": "def merge_sorted(a, b):\n    result = []\n    i, j = 0, 0\n    while i < len(a) and j < len(b):\n        if a[i] <= b[j]:\n            result.append(a[i])\n            i += 1\n        else:\n            result.append(b[j])\n            j += 1\n    result.extend(a[i:])\n    result.extend(a[j:])\n    return result",
        "tests": [
            "assert merge_sorted([1, 3, 5], [2, 4, 6]) == [1, 2, 3, 4, 5, 6]",
            "assert merge_sorted([], [1, 2]) == [1, 2]",
            "assert merge_sorted([1, 2], []) == [1, 2]",
            "assert merge_sorted([1], [2, 3, 4]) == [1, 2, 3, 4]",
            "assert merge_sorted([], []) == []",
        ],
        "difficulty": 2,
    },

    # ── 12. String interleaving with index error ─────────────────────
    {
        "id": "med_interleave_strings",
        "description": "The function `interleave(s1, s2)` should interleave two strings character by character. If one is longer, append the remaining characters. interleave('abc', 'XY') should return 'aXbYc'.",
        "buggy_code": "def interleave(s1, s2):\n    result = []\n    for i in range(max(len(s1), len(s2))):\n        if i < len(s1):\n            result.append(s1[i])\n        if i < len(s2):\n            result.append(s2[i])\n    return result",
        "tests": [
            "assert interleave('abc', 'XY') == 'aXbYc'",
            "assert interleave('', 'xyz') == 'xyz'",
            "assert interleave('ab', 'cd') == 'acbd'",
            "assert interleave('a', '') == 'a'",
            "assert interleave('', '') == ''",
        ],
        "difficulty": 2,
    },

    # ── 13. Counter class with __eq__ and state mutation ──────────────
    {
        "id": "med_counter_reset",
        "description": "The `Counter` class counts up from 0. `increment()` adds 1, `get()` returns current count, `reset()` sets count back to 0. The reset method does not work correctly.",
        "buggy_code": "class Counter:\n    def __init__(self):\n        self.count = 0\n\n    def increment(self):\n        self.count += 1\n\n    def get(self):\n        return self.count\n\n    def reset(self):\n        count = 0",
        "tests": [
            "c = Counter(); c.increment(); c.increment(); assert c.get() == 2",
            "c = Counter(); c.increment(); c.reset(); assert c.get() == 0",
            "c = Counter(); c.reset(); assert c.get() == 0",
            "c = Counter(); c.increment(); c.increment(); c.reset(); c.increment(); assert c.get() == 1",
        ],
        "difficulty": 2,
    },

    # ── 14. Recursive tree depth with off-by-one ─────────────────────
    {
        "id": "med_nested_depth",
        "description": "The function `max_nesting(lst)` should return the maximum nesting depth of a list. A flat list like [1, 2] has depth 1. [[1], 2] has depth 2. [[[1]]] has depth 3.",
        "buggy_code": "def max_nesting(lst):\n    if not isinstance(lst, list):\n        return 0\n    if not lst:\n        return 1\n    max_child = 0\n    for item in lst:\n        child_depth = max_nesting(item)\n        if child_depth > max_child:\n            max_child = child_depth\n    return max_child",
        "tests": [
            "assert max_nesting([1, 2, 3]) == 1",
            "assert max_nesting([[1], 2]) == 2",
            "assert max_nesting([[[1]]]) == 3",
            "assert max_nesting([]) == 1",
            "assert max_nesting([1, [2, [3, [4]]]]) == 4",
        ],
        "difficulty": 3,
    },

    # ── 15. zip truncation with dict construction ────────────────────
    {
        "id": "med_zip_truncation",
        "description": "The function `make_mapping(keys, values, default=None)` should return a dict mapping each key to its corresponding value. If there are more keys than values, the extra keys should map to `default`. If there are more values than keys, the extra values are ignored.",
        "buggy_code": "def make_mapping(keys, values, default=None):\n    return dict(zip(keys, values))",
        "tests": [
            "assert make_mapping(['a', 'b', 'c'], [1, 2, 3]) == {'a': 1, 'b': 2, 'c': 3}",
            "assert make_mapping(['a', 'b', 'c'], [1]) == {'a': 1, 'b': None, 'c': None}",
            "assert make_mapping(['a'], [1, 2, 3]) == {'a': 1}",
            "assert make_mapping(['x', 'y'], [10], default=-1) == {'x': 10, 'y': -1}",
            "assert make_mapping([], [1, 2]) == {}",
        ],
        "difficulty": 2,
    },
]

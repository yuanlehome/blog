---
title: leetcode刷题系列之数据结构设计
slug: leetcode
date: '2021-07-10'
tags: []
status: published
source_url: 'https://yuanlehome.github.io/blog_v0/GRrK24ZqnqkWXLK0/'
source_author: Liu Y.L.
imported_at: '2026-01-28T16:06:58.408Z'
source:
  title: yuanlehome.github.io
  url: 'https://yuanlehome.github.io/blog_v0/GRrK24ZqnqkWXLK0/'
updated: '2021-07-10'
---

# leetcode刷题系列之数据结构设计

\*\*发表于 2021-07-10 \*\*分类于 [leetcode](/blog_v0/categories/leetcode/) \*\*阅读次数： 2\
\*\*本文字数： 36k \*\*阅读时长 ≈ 32 分钟

这篇文章是`leetcode`刷题系列的第`7`部分——数据结构设计。

`leetcode`刷题系列其它文章组织如下：

`1`. [数组](https://yuanlehome.github.io/qD0F57Dbj7HjnZou/)

`2`. [链表](https://yuanlehome.github.io/S07PSuYxoZ6CPova/)

`3`. [字符串](https://yuanlehome.github.io/LZqUbK3Z1CXKja4I/)

`4`. [二叉树](https://yuanlehome.github.io/B90hHtDrYEYJD3xv/)

`5`. [队列和栈](https://yuanlehome.github.io/fhQPnKWa9qDDelG3/)

`6`. [动态规划](https://yuanlehome.github.io/RT66rbCYdVwFEsD8/)

`7`. [数据结构设计](https://yuanlehome.github.io/GRrK24ZqnqkWXLK0/)

`8`. [刷题小知识点](https://yuanlehome.github.io/MK80vfKBcuYfGiyp/)

[]()

##### [](#622-Design-Circular-Queue '622. Design Circular Queue')[622. Design Circular Queue](https://leetcode.com/problems/design-circular-queue/)

> 设计你的循环队列实现。 循环队列是一种线性数据结构，其操作表现基于`FIFO`（先进先出）原则并且队尾被连接在队首之后以形成一个循环。它也被称为“环形缓冲器”。
>
> 循环队列的一个好处是我们可以利用这个队列之前用过的空间。在一个普通队列里，一旦一个队列满了，我们就不能插入下一个元素，即使在队列前面仍有空间。但是使用循环队列，我们能使用这些空间去存储新的值。
>
> 你的实现应该支持如下操作：
>
> |                                                |        |
> | ---------------------------------------------- | ------ |
> | \`\`\`                                         |        |
> | 1                                              |        |
> | 2                                              |        |
> | 3                                              |        |
> | 4                                              |        |
> | 5                                              |        |
> | 6                                              |        |
> | 7                                              |        |
> | 8                                              |        |
> | 9                                              |        |
> | 10                                             |        |
> | 11                                             |        |
> | 12                                             |        |
> | 13                                             |        |
> | 14                                             |        |
> | \`\`\`                                         | \`\`\` |
> | MyCircularQueue(k)                             |        |
> | 构造器, 设置队列长度为 k                       |        |
> | Front                                          |        |
> | 从队首获取元素, 如果队列为空，返回 -1          |        |
> | Rear                                           |        |
> | 获取队尾元素, 如果队列为空，返回 -1            |        |
> | enQueue(value)                                 |        |
> | 向循环队列插入一个元素, 如果成功插入则返回真   |        |
> | deQueue()                                      |        |
> | 从循环队列中删除一个元素, 如果成功删除则返回真 |        |
> | isEmpty()                                      |        |
> | 检查循环队列是否为空                           |        |
> | isFull()                                       |        |
> | 检查循环队列是否已满                           |        |
>
> ```|
>
> ```

|        |        |
| ------ | ------ |
| \`\`\` |        |
| 1      |        |
| 2      |        |
| 3      |        |
| 4      |        |
| 5      |        |
| 6      |        |
| 7      |        |
| 8      |        |
| 9      |        |
| 10     |        |
| 11     |        |
| 12     |        |
| 13     |        |
| 14     |        |
| 15     |        |
| 16     |        |
| 17     |        |
| 18     |        |
| 19     |        |
| 20     |        |
| 21     |        |
| 22     |        |
| 23     |        |
| 24     |        |
| 25     |        |
| 26     |        |
| 27     |        |
| 28     |        |
| 29     |        |
| 30     |        |
| 31     |        |
| 32     |        |
| 33     |        |
| 34     |        |
| 35     |        |
| 36     |        |
| 37     |        |
| 38     |        |
| 39     |        |
| 40     |        |
| 41     |        |
| 42     |        |
| 43     |        |
| 44     |        |
| 45     |        |
| 46     |        |
| 47     |        |
| 48     |        |
| 49     |        |
| 50     |        |
| 51     |        |
| 52     |        |
| 53     |        |
| 54     |        |
| 55     |        |
| 56     |        |
| 57     |        |
| 58     |        |
| 59     |        |
| 60     |        |
| 61     |        |
| 62     |        |
| 63     |        |
| 64     |        |
| 65     |        |
| 66     |        |
| 67     |        |
| 68     |        |
| 69     |        |
| 70     |        |
| 71     |        |
| 72     |        |
| 73     |        |
| 74     |        |
| \`\`\` | \`\`\` |

class MyCircularQueue {
private:
int \_head;
int \_tail;
int \_size;
vector<int> \_data;

public:

```c
MyCircularQueue(int k) : _head(-1), _tail(-1), _size(0) {

    _data.resize(k);
}

bool enQueue(int value) {

    if(isFull()) {
        return false;
    }


    if(++_tail == _data.size()) {
        _tail = 0;
    }
    _data[_tail] = value;
    _size++;
    return true;
}

bool deQueue() {

    if(isEmpty()) {
        return false;
    }


    _head++;

    if(_head == _data.size()) {
        _head = 0;
    }
    _size--;
    return true;
}

int Front() const {
    if(isEmpty()) {
        return -1;
    }
    return _head + 1 == _data.size() ? _data[0] : _data[_head + 1];
}

int Rear() const {
    return isEmpty() ? -1 : _data[_tail];
}

int size() const {
    return _size;
}

bool isEmpty() const {
    return _size == 0;
}

bool isFull() const {
    return _size == _data.size();
}
```

};

````|

##### [](#146-LRU-Cache "146. LRU Cache")[146. LRU Cache](https://leetcode.com/problems/lru-cache/)

> 运用你所掌握的数据结构，设计和实现一个`LRU`(最近最少使用) 缓存机制 。
>
> **参考链接**：
>
> - [从 LRU Cache 带你看面试的本质](https://mp.weixin.qq.com/s?__biz=MzIzNDQ3MzgxMw==\&mid=2247483929\&idx=1\&sn=fda81057c47d376917ed142b2661f63a\&chksm=e8f49223df831b35deb2e5316caddc241b4aa4bb58e8c66c906bdacfd695aca53aca86a5b173\&mpshare=1\&scene=23\&srcid=0316bjmcFhe5xBzFM5mVMehZ\&sharer_sharetime=1622459011412\&sharer_shareid=7cbdd205bcb5ea7a7912ce1a62c48cda#rd)
> - [缓存淘汰算法的实现与应用介绍（LRU、LFU）](https://mp.weixin.qq.com/s?__biz=Mzk0NTE5MTcxNQ==\&mid=2247483722\&idx=1\&sn=4f5ff638f9e020ad7ee8a3ba2595b8e6\&chksm=c3186f06f46fe610a98fcc65e397f22d6d39a25ed32715ae313ec45a0f7f19368c36ec3175bb\&mpshare=1\&scene=23\&srcid=0303Bs1lcVsbvfREx8q7YaqB\&sharer_sharetime=1622459025098\&sharer_shareid=7cbdd205bcb5ea7a7912ce1a62c48cda#rd)
> - [我竟然跪在了LRU，好亏奥！](https://mp.weixin.qq.com/s?__biz=MzA4NDE4MzY2MA==\&mid=2647523909\&idx=1\&sn=fc26b334afcd3b12a130905043e58f20\&chksm=87d1bd46b0a634505a619566b7cd56c4f69bec7793d3f6794cef1207c3c40bdc6916da608089\&mpshare=1\&scene=23\&srcid=1226uXTA9n8lmyoSP4zAxDly\&sharer_sharetime=1622459037018\&sharer_shareid=7cbdd205bcb5ea7a7912ce1a62c48cda#rd)
>
> 实现`LRUCache`类：
>
> |                     |                                                                                                                                                                                                                                           |
> | ------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
> | ```
> 1
> 2
> 3
> 4
> 5
> 6
> ``` | ```
> LRUCache(int capacity)
> 以正整数作为容量 capacity 初始化 LRU 缓存
> int get(int key)
> 如果关键字 key 存在于缓存中，则返回关键字的值, 否则返回 -1
> void put(int key, int value)
> 如果关键字已经存在, 则变更其数据值: 如果关键字不存在, 则插入该组「关键字-值」, 当缓存容量达到上限时, 它应该在写入新数据之前删除最久未使用的数据值, 从而为新的数据值留出空间
> ``` |

![Image](dtter.jpg) ![Image](2341123.jpg)

|                                                                                                                  |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| ---------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| ```
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
31
32
33
34
35
36
37
38
``` | ```
template <typename T>
class LRUCache {
private:
    typedef T value_type;
    typedef int key_type;
    typedef typename list<pair<key_type, value_type>>::iterator iterator_to_node;
    unordered_map<key_type, iterator_to_node> _key2item;
    list<pair<key_type, value_type>> _items;
    size_t _capacity;

public:
    LRUCache(size_t capacity) : _capacity(capacity) {};
    ~LRUCache() = default;

    value_type get(key_type key) {
        value_type value;
        if(_key2item.count(key)) {
            value = _key2item[key]->second;
            _items.splice(_items.begin(), _items, _key2item[key]);
        }
        return value;
    }

    void put(key_type key, value_type value) {
        if(_key2item.count(key)) {
            _key2item[key]->second = value;
            _items.splice(_items.begin(), _items, _key2item[key]);
        }
        else {
            if(_capacity <= _items.size()) {
                _key2item.erase(_items.back().first);
                _items.pop_back();
            }
             _items.emplace_front(key, value);
            _key2item[key] = _items.begin();
        }
    }
}
``` |

|                                                                                                                           |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| ------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ```
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
31
32
33
34
35
36
37
38
39
40
41
``` | ```

template <typename T>
class LRUCache {
private:
    typedef T value_type;
    typedef int key_type;
    typedef typename list<pair<key_type, value_type>>::iterator iterator_to_node;
    size_t _capacity;
    unordered_map<key_type, iterator_to_node> _keyToItem;
    list<pair<key_type, value_type>> _itemList;

public:
    LRUCache(size_t capacity) : _capacity(capacity) {}
    ~LRUCache() {}

    value_type get(key_type key) {
        value_type value;
        if(_keyToItem.count(key)) {
            value = _keyToItem[key]->second;

        	put(key, value);
        }

        return value;
    }

    void put(key_type key, value_type val) {

        if(_keyToItem.count(key)) {
            _itemList.erase(_keyToItem[key]);
        }
        else if(_capacity <= _itemList.size()) {

            _keyToItem.erase(_itemList.back().first);
            _itemList.pop_back();
        }

        _itemList.emplace_front(key, val);
        _keyToItem[key] = _itemList.begin();
    }
};
``` |

|                                                                                                                                                                                                                                                                                                         |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ```
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
31
32
33
34
35
36
37
38
39
40
41
42
43
44
45
46
47
48
49
50
51
52
53
54
55
56
57
58
59
60
61
62
63
64
65
66
67
68
69
70
71
72
73
74
75
76
77
78
79
80
81
82
83
84
85
86
87
88
89
90
91
92
93
94
95
96
97
98
99
``` | ```

template <typename T>
struct Node {
    T _value;
    Node* _prev;
    Node* _next;
    Node(const T& value) : _value(value), _prev(nullptr), _next(nullptr) {}
};

template <typename T>
class DoubleLinkedList {
public:
    typedef T value_type;
    typedef Node<value_type>* link_type;

private:
    link_type _node;
    size_t _size;

public:
    DoubleLinkedList() : _node(nullptr), _size(0) {
        _node = new Node<value_type>(value_type());
        _node->_next = _node;
        _node->_prev = _node;
    }

    ~DoubleLinkedList() {
        while(_size > 0) {
            pop_front();
        }
        delete _node;
    }

    link_type begin() const { return _node->_next; }
    link_type end() const { return _node; }
    value_type& front() const { return _node->_next->_value; }
    value_type& back() const { return _node->_prev->_value; }
    size_t size() const { return _size; }

    void erase(link_type node) {
        _size--;
        node->_prev->_next = node->_next;
        node->_next->_prev = node->_prev;
        delete node;
    }
    void pop_front() { erase(_node->_next); }
    void pop_back() { erase(_node->_prev); }

    void insert(link_type pos, value_type value) {
        _size++;
        link_type node = new Node<value_type>(value);
        pos->_prev->_next = node;
        node->_prev = pos->_prev;
        pos->_prev = node;
        node->_next = pos;
    }
    void push_front(value_type value) { insert(_node->_next, value); }
    void push_back(value_type value) { insert(_node, value); }
};

template <typename T>
class LRUCache {
private:
    typedef T value_type;
    typedef typename DoubleLinkedList<pair<int, value_type>>::link_type iterator_to_node;
    size_t _capacity;
    unordered_map<int, iterator_to_node> _keyToItem;
    DoubleLinkedList<pair<int, value_type>> _itemList;

public:
    LRUCache(int capacity) : _capacity(capacity) {}
    ~LRUCache() {}

    value_type get(int key) {
        if(_keyToItem.count(key) == 0) {
            return -1;
        }
        value_type res = _keyToItem[key]->_value.second;

        put(key, res);

        return res;
    }

    void put(int key, value_type val) {

        if(_keyToItem.count(key)) {
            _itemList.erase(_keyToItem[key]);
        }
        else if(_capacity <= _itemList.size()) {

            _keyToItem.erase(_itemList.back().first);
            _itemList.pop_back();
        }

        _itemList.push_front(make_pair(key, val));
        _keyToItem[key] = _itemList.begin();
    }
};
``` |

##### [](#460-LFU-Cache "460. LFU Cache")[460. LFU Cache](https://leetcode.com/problems/lfu-cache/)

> 请你为最不经常使用（`LFU`）缓存算法设计并实现数据结构。
>
> 实现`LFUCache`类：
>
> |                     |                                                                                                                                                                                                                                               |
> | ------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
> | ```
> 1
> 2
> 3
> 4
> 5
> 6
> ``` | ```
> LFUCache(int capacity)
> 用数据结构的容量 capacity 初始化对象
> int get(int key)
> 如果键存在于缓存中, 则获取键的值, 否则返回 -1
> void put(int key, int value)
> 如果键已存在, 则变更其值; 如果键不存在, 请插入键值对, 当缓存达到其容量时, 则应该在插入新项之前, 使最不经常使用的项无效, 在此问题中，当存在平局（即两个或更多个键具有相同使用频率）时，应该去除最久未使用的键
> ``` |
>
> 注意「项的使用次数」就是自插入该项以来对其调用`get`和`put`函数的次数之和。使用次数会在对应项被移除后置为`0`。
>
> 为了确定最不常使用的键，可以为缓存中的每个键维护一个使用计数器 。使用计数最小的键是最久未使用的键。当一个键首次插入到缓存中时，它的使用计数器被设置为`1` (由于`put`操作)。对缓存中的键执行`get`或`put`操作，使用计数器的值将会递增。

![Image](dfgdf.jpg)

|                                                                                                                                                                                                                                          |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ```
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
31
32
33
34
35
36
37
38
39
40
41
42
43
44
45
46
47
48
49
50
51
52
53
54
55
56
57
58
59
60
61
62
63
64
65
66
67
68
69
70
71
72
73
74
75
76
77
78
``` | ```

template <typename T>
struct Node {
    typedef T value_type;
    int _key;
    value_type _value;
    size_t _freq;
    Node(int key, value_type value, size_t freq) : _key(key), _value(value), _freq(freq) {}
};

template <typename T>
class LFUCache {
private:
    typedef T value_type;
    typedef typename list<Node<value_type>>::iterator iterator_to_node;
    size_t _capacity;
    size_t _minFreq;
    unordered_map<int, iterator_to_node> _keyToItem;
    unordered_map<size_t, list<Node<value_type>>> _freqToKeys;

public:
    LFUCache(size_t capacity) : _capacity(capacity), _minFreq(0) {}
    ~LFUCache() {}

    value_type get(int key) {
        if(_keyToItem.count(key) == 0) {
            return value_type();
        }

        increaseFreq(key);
        return _keyToItem[key]->_value;
    }

    void put(int key, value_type value) {

        if(_keyToItem.count(key)) {
            _keyToItem[key]->_value = value;
            increaseFreq(key);
            return;
        }

        if(_capacity <= _keyToItem.size()) {
            auto lfu_key = _freqToKeys[_minFreq].back()._key;
            _freqToKeys[_minFreq].pop_back();
            _keyToItem.erase(lfu_key);

            if(_freqToKeys[_minFreq].empty()) {
                _freqToKeys.erase(_minFreq);
            }

        }

        _freqToKeys[1].push_front(Node<value_type>(key, value, 1));
        _keyToItem[key] = _freqToKeys[1].begin();

        _minFreq = 1;
    }

private:

    void increaseFreq(int key) {

        size_t theFreq = _keyToItem[key]->_freq;
        value_type theValue = _keyToItem[key]->_value;

        _freqToKeys[theFreq].erase(_keyToItem[key]);
        _freqToKeys[theFreq + 1].push_front(Node<value_type>(key, theValue, theFreq + 1));
        _keyToItem[key] = _freqToKeys[theFreq + 1].begin();

        if(_freqToKeys[theFreq].empty()) {
            _freqToKeys.erase(theFreq);

            if(theFreq == _minFreq) {
                _minFreq++;
            }
        }
    }
};
``` |

##### [](#208-Implement-Trie-Prefix-Tree "208. Implement Trie (Prefix Tree)")[208. Implement Trie (Prefix Tree)](https://leetcode.com/problems/implement-trie-prefix-tree/)

> `Trie`或者说前缀树是一种树形数据结构，用于高效地存储和检索字符串数据集中的键。这一数据结构有相当多的应用情景，例如自动补完和拼写检查。
>
> 请你实现`Trie`类：
>
> - `Trie()` 初始化前缀树对象。
> - `void insert(String word)` 向前缀树中插入字符串`word`。
> - `boolean search(String word)` 如果字符串`word`在前缀树中，返回`true`（即，在检索之前已经插入）；否则，返回`false`。
> - `boolean startsWith(String prefix)` 如果之前已经插入的字符串`word`的前缀之一为`prefix`，返回`true`；否则，返回`false`。

|                                                                                                                                             |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| ------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ```
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
31
32
33
34
35
36
37
38
39
40
41
42
43
44
45
46
47
``` | ```
class Trie {
private:
    vector<Trie*> child;
    bool isEnd;

public:
    Trie() : child(26), isEnd(false) {}
    ~Trie() {
        for(auto node : child) {
            if(node) {
                delete node;
            }
        }
    }

    void insert(string word) {
        Trie* node = this;
        for(char c : word) {
            if(!node->child[c - 'a']) {
                node->child[c] = new Trie;
            }
            node = node->child[c - 'a'];
        }
        node->isEnd = true;
    }

    bool search(string word) {
        Trie* node = searchPrefix(word);
        return node && node->isEnd;
    }

    bool startsWith(string prefix) {
        return searchPrefix(prefix);
    }

private:
    Trie* searchPrefix(string prefix) {
        Trie* node = this;
        for(char c : prefix) {
            if(!node->child[c - 'a']) {
                return nullptr;
            }
            node = node->child[c - 'a'];
        }
        return node;
    }
};
``` |

##### [](#295-Find-Median-from-Data-Stream "295. Find Median from Data Stream")[295. Find Median from Data Stream](https://leetcode.com/problems/find-median-from-data-stream/)

> The **median** is the middle value in an ordered integer list. If the size of the list is even, there is no middle value and the median is the mean of the two middle values.
>
> - For example, for `arr = [2,3,4]`, the median is `3`.
>
> - For example, for `arr = [2,3]`, the median is `(2 + 3) / 2 = 2.5`.
>
>   Implement the `MedianFinder` class:
>
> - `MedianFinder()` initializes the `MedianFinder` object.
>
> - `void addNum(int num)` adds the integer `num` from the data stream to the data structure.
>
> - `double findMedian()` returns the median of all elements so far. Answers within `10-5` of the actual answer will be accepted.
>
> **Example:**
>
> |                                       |                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
> | ------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
> | ```
> 1
> 2
> 3
> 4
> 5
> 6
> 7
> 8
> 9
> 10
> 11
> 12
> 13
> ``` | ```
> Input
> ["MedianFinder", "addNum", "addNum", "findMedian", "addNum", "findMedian"]
> [[], [1], [2], [], [3], []]
> Output
> [null, null, null, 1.5, null, 2.0]
>
> Explanation
> MedianFinder medianFinder = new MedianFinder();
> medianFinder.addNum(1);    // arr = [1]
> medianFinder.addNum(2);    // arr = [1, 2]
> medianFinder.findMedian(); // return 1.5 (i.e., (1 + 2) / 2)
> medianFinder.addNum(3);    // arr[1, 2, 3]
> medianFinder.findMedian(); // return 2.0
> ``` |
>
> **Constraints:**
>
> - `-10^5 <= num <= 10^5`
> - There will be at least one element in the data structure before calling `findMedian`.
> - At most `5 * 10^4` calls will be made to `addNum` and `findMedian`.
>
> **Follow up:**
>
> - If all integer numbers from the stream are in the range `[0, 100]`, how would you optimize your solution?
> - If `99%` of all integer numbers from the stream are in the range `[0, 100]`, how would you optimize your solution?

|                                                                                                                           |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| ------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ```
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
31
32
33
34
35
36
37
38
39
40
41
``` | ```
class MedianFinder
{
private:
    priority_queue<int> maxTop;
    priority_queue<int, vector<int>, greater<int>> minTop;

public:
    MedianFinder() {}

    void addNum(int num)
    {
        if(minTop.empty() || minTop.top() <= num)
            minTop.push(num);
        else
            maxTop.push(num);

        balance();
    }

    double findMedian()
    {
        if(minTop.size() == maxTop.size())
            return (double(minTop.top()) + double(maxTop.top())) * 0.5;
        return minTop.top();
    }

private:
    void balance()
    {
        if(minTop.size() < maxTop.size())
        {
            minTop.push(maxTop.top());
            maxTop.pop();
        }
        if(minTop.size() > maxTop.size() + 1)
        {
            maxTop.push(minTop.top());
            minTop.pop();
        }
    }
};
``` |

|                                                                                          |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| ---------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| ```
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
``` | ```

class MedianFinder {
private:
    priority_queue<int> small;
    priority_queue<int, vector<int>, greater<int>> large;

public:
    MedianFinder() {}

    void addNum(int num) {

        if(small.size() == large.size()) {
            large.push(num);
            small.push(large.top());
            large.pop();
        }
        else {
            small.push(num);
            large.push(small.top());
            small.pop();
        }
    }

    double findMedian() {
        if(small.size() != large.size()) {
            return small.top();
        }
        return (double(small.top()) + double(large.top())) * 0.5;
    }
};
``` |

##### [](#170-两数之和-III-数据结构设计 "170. 两数之和 III - 数据结构设计")[170. 两数之和 III - 数据结构设计](https://leetcode-cn.com/problems/two-sum-iii-data-structure-design/)

> 设计一个`TwoSum`类，拥有两个`API`：
>
> |                       |                                                                                       |
> | --------------------- | ------------------------------------------------------------------------------------- |
> | ```
> 1
> 2
> 3
> 4
> 5
> 6
> 7
> ``` | ```
> >class TwoSum {
> >public:
> >
> >void add(int number);
> >
> >bool find(int value);
> >}
> ``` |

|                                                                                 |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| ------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ```
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
``` | ```

class TwoSum {
private:
    unordered_map<long, int> mapping;

public:

    void add(int number) {
        mapping[number]++;
    }


    bool find(int value) {
        for(auto&& [first, _] : mapping) {
            long second = long(value) - first;

            if(second == first && mapping[first] > 1) {
                return true;
            }

            if(second != first && mapping.count(second) && mapping[second] > 0) {
                return true;
            }
        }
        return false;
    }
};
``` |

|                                                               |                                                                                                                                                                                                                                                                                                                                                      |
| ------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ```
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
``` | ```

class TwoSum {
private:
    unordered_set<int> allSum;
    vector<int> nums;

public:

    void add(int number) {

        for(int num : nums) {
            allSum.insert(num + number);
        }
        nums.push_back(number);
    }


    bool find(int value) {
        return allSum.count(value);
    }
};
``` |

##### [](#155-Min-Stack "155. Min Stack")[155. Min Stack](https://leetcode.com/problems/min-stack/)

> 设计一个支持`push`，`pop`，`top`操作，并能在常数时间内检索到最小元素的栈。
>
> `push(x)` —— 将元素`x`推入栈中。\
> `pop()` —— 删除栈顶的元素。\
> `top()` —— 获取栈顶元素。\
> `getMin()` —— 检索栈中的最小元素。

|                                                                                          |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| ---------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| ```
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
``` | ```
class MinStack {
private:

    stack<pair<int, int>> _data;

public:

    MinStack() {}

    void push(int x) {
        if(_data.empty()) {
            _data.push({x, x});
        }
        else {
        	_data.push({x, min(x, _data.top().second)});
        }
    }

    void pop() {
        _data.pop();
    }

    int top() {
        return _data.top().first;
    }

    int getMin() {
        return _data.top().second;
    }
};
``` |

##### [](#895-Maximum-Frequency-Stack "895. Maximum Frequency Stack")[895. Maximum Frequency Stack](https://leetcode.com/problems/maximum-frequency-stack/)

> 实现`FreqStack`，模拟类似栈的数据结构的操作的一个类。
>
> `FreqStack`有两个函数：
>
> - `push(int x)`，将整数`x`推入栈中。
>
> - `pop()`，它移除并返回栈中出现最频繁的元素。
>
>   如果最频繁的元素不只一个，则移除并返回最接近栈顶的元素。
>
>   提示：
>
> - 对`FreqStack.push(int x)`的调用中`0 <= x <= 10^9`。
>
> - 如果栈的元素数目为`0`，则保证不会调用`FreqStack.pop()`。
>
> **示例**：
>
> ![img](8c5d16af06b2bbf15ac75dad30898e99c0b19b83d433b303a4f0fb8ac885387b.jpg)
>
> |                                       |                                                                                                                                                                                                          |
> | ------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
> | ```
> 1
> 2
> 3
> 4
> 5
> 6
> 7
> 8
> 9
> 10
> 11
> 12
> 13
> ``` | ```
> 比如执行六次 push 操作后，栈自底向上为 [5,7,5,7,4,5]
> 然后:
> pop() -> 返回 5，因为 5 是出现频率最高的
> 栈变成 [5,7,5,7,4]
>
> pop() -> 返回 7，因为 5 和 7 都是频率最高的，但 7 最接近栈顶
> 栈变成 [5,7,5,4]
>
> pop() -> 返回 5
> 栈变成 [5,7,4]
>
> pop() -> 返回 4
> 栈变成 [5,7]
> ``` |

|                                                                                                |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| ---------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ```
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
31
32
``` | ```
class FreqStack {
private:
    size_t _maxFreq;
    unordered_map<int, size_t> _valToFreq;
    unordered_map<size_t, stack<int>> _freqToVals;

public:
    FreqStack() : _maxFreq(0) {}

    void push(int val) {
        _valToFreq[val]++;
        int freq = _valToFreq[val];
        if(_maxFreq < freq) {
            _maxFreq = freq;
        }
        _freqToVals[freq].push(val);
    }

    int pop() {
        int res = _freqToVals[_maxFreq].top();
        _freqToVals[_maxFreq].pop();
        _valToFreq[res]--;
        if(_valToFreq[res] == 0) {
            _valToFreq.erase(res);
        }
        if(_freqToVals[_maxFreq].empty()) {
            _freqToVals.erase(_maxFreq);
            _maxFreq--;
        }
        return res;
    }
};
``` |

##### [](#232-Implement-Queue-using-Stacks "232. Implement Queue using Stacks")[232. Implement Queue using Stacks](https://leetcode.com/problems/implement-queue-using-stacks/)

> 请你仅使用两个栈实现先入先出队列。队列应当支持一般队列支持的所有操作：
>
> 实现`MyQueue`类：
>
> `void push(int x)`：将元素x推到队列的末尾；\
> `int pop()`：从队列的开头移除并返回元素；\
> `int peek()`：返回队列开头的元素；\
> `bool empty()`：如果队列为空，返回`true`；否则，返回`false`。
>
> ![img](2.jpg)

|                                                                                                                                                   |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| ------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ```
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
31
32
33
34
35
36
37
38
39
40
41
42
43
44
45
46
47
48
49
``` | ```

class MyQueue {
private:
    stack<int> _front;
    stack<int> _back;

public:
    MyQueue() {}

    void push(int x) {

        _back.push(x);
    }

    int pop() {


        if(_front.empty()) {
            moveData();
        }

        int res = _front.top();
        _front.pop();
        return res;
    }

    int peek() {

        if(_front.empty()) {
            moveData();
        }

        return _front.top();
    }

    bool empty() {
        return _front.empty() && _back.empty();
    }

private:
    void moveData() {


        while(!_back.empty()) {
            _front.push(move(_back.top()));
            _back.pop();
        }
    }
};
``` |

##### [](#225-Implement-Stack-using-Queues "225. Implement Stack using Queues")[225. Implement Stack using Queues](https://leetcode.com/problems/implement-stack-using-queues/)

> 请你仅使用两个队列实现一个后入先出的栈，并支持普通队列的全部四种操作。
>
> 实现`MyStack`类：
>
> `void push(int x)`：将元素`x`压入栈顶；\
> `int pop()`：移除并返回栈顶元素；\
> `int top()`：返回栈顶元素；\
> `bool empty()`：如果栈是空的，返回`true`；否则，返回`false`。

|                                                                                                                              |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| ---------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ```
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
31
32
33
34
35
36
37
38
39
40
41
42
``` | ```

class MyStack {
private:
    queue<int> _queue;
    int _top;

public:
    MyStack() {}

    void push(int x) {

        _top = x;
        _queue.push(x);
    }

    int pop() {
        int sz = _queue.size();
        while(sz-- > 1) {
            _top = _queue.front();
            _queue.pop();
            _queue.push(_top);
        }
        int res = _queue.front();
        _queue.pop();
        return res;
    }

    int top() {
        return _top;
    }

    bool empty() {
        return _queue.empty();
    }
};
``` |

|                                                                                             |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| ------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| ```
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
31
``` | ```

class MyStack {
private:
    queue<int> _queue;

public:
    MyStack() {}

    void push(int x) {
        int sz = _queue.size();
        _queue.push(x);
        while(sz-- > 0) {
            _queue.push(_queue.front());
            _queue.pop();
        }
    }

    int pop() {
        int res = _queue.front();
        _queue.pop();
        return res;
    }

    int top() {
        return _queue.front();
    }

    bool empty() {
        return _queue.empty();
    }
};
``` |

##### [](#432-All-O-one-Data-Structure-https-leetcode-com-problems-all-oone-data-structure "\[432. All O`one Data Structure](https://leetcode.com/problems/all-oone-data-structure/)")\[432. All O\`one Data Structure]\([https://leetcode.com/problems/all-oone-data-structure/](https://leetcode.com/problems/all-oone-data-structure/))

> 请你实现一个数据结构支持以下操作：
>
> - `Inc(key)` 插入一个新的值为`1`的`key`。或者使一个存在的`key`增加`1`，保证`key`不为空字符串。
> - `Dec(key)` 如果这个`key`的值是`1`，那么把他从数据结构中移除掉。否则使一个存在的`key`值减`1`。如果这个`key`不存在，这个函数不做任何事情。`key`保证不为空字符串。
> - `GetMaxKey()` 返回`key`中值最大的任意一个。如果没有元素存在，返回一个空字符串`""`。
> - `GetMinKey()` 返回`key`中值最小的任意一个。如果没有元素存在，返回一个空字符串`""`。

|                                                                                                                                                                                                                                                               |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ```
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
31
32
33
34
35
36
37
38
39
40
41
42
43
44
45
46
47
48
49
50
51
52
53
54
55
56
57
58
59
60
61
62
63
64
65
66
67
68
69
70
71
72
73
74
75
76
77
78
79
80
81
82
83
84
85
``` | ```

class AllOne
{
private:
    typedef string key_type;
    typedef size_t value_type;
    unordered_map<key_type, list<pair<value_type, unordered_set<key_type>>>::iterator> _keyToNode;

    list<pair<value_type, unordered_set<key_type>>> _nodes;

public:
    AllOne() {}

    void inc(string key)
    {
        if(_keyToNode.count(key))
        {
            auto it = _keyToNode[key];
            auto next_it = next(it);
            if(next_it == _nodes.end() || next_it->first != (it->first + 1))
            {
                _keyToNode[key] = _nodes.insert(next_it, {it->first + 1, {key}});
            }
            else
            {
                next_it->second.insert(key);
                _keyToNode[key] = next_it;
            }
            it->second.erase(key);
            if(it->second.empty())
                _nodes.erase(it);
        }
        else
        {
            if(_nodes.empty() || _nodes.begin()->first != 1)
            {
                _keyToNode[key] = _nodes.insert(_nodes.begin(), {1, {key}});
            }
            else
            {
                _nodes.begin()->second.insert(key);
                _keyToNode[key] = _nodes.begin();
            }
        }
    }

    void dec(string key)
    {
        if(_keyToNode.count(key))
        {
            auto it = _keyToNode[key];
            auto prev_it = prev(it);
            if(it->first > 1)
            {
                if(it == _nodes.begin() || prev_it->first != (it->first - 1))
                {
                    _keyToNode[key] = _nodes.insert(it, {it->first - 1, {key}});
                }
                else
                {
                    prev_it->second.insert(key);
                    _keyToNode[key] = prev_it;
                }
            }
            else
            {
                _keyToNode.erase(key);
            }
            it->second.erase(key);
            if(it->second.empty())
                _nodes.erase(it);
        }
    }

    string getMaxKey()
    {
        return _nodes.empty() ? "" : *(_nodes.rbegin()->second.begin());
    }

    string getMinKey()
    {
        return _nodes.empty() ? "" : *(_nodes.begin()->second.begin());
    }
};
``` |

##### [](#707-Design-Linked-List "707. Design Linked List")[707. Design Linked List](https://leetcode.com/problems/design-linked-list/)

> 设计链表的实现。您可以选择使用单链表或双链表。单链表中的节点应该具有两个属性：`val`和`next`。`val`是当前节点的值，`next`是指向下一个节点的指针/引用。如果要使用双向链表，则还需要一个属性`prev`以指示链表中的上一个节点。假设链表中的所有节点都是`0 - index`的。
>
> 在链表类中实现这些功能：
>
> - `get(index)`：获取链表中第`index`个节点的值。如果索引无效，则返回`-1`。
> - `addAtHead(val)`：在链表的第一个元素之前添加一个值为`val`的节点。插入后，新节点将成为链表的第一个节点。
> - `addAtTail(val)`：将值为`val`的节点追加到链表的最后一个元素。
> - `addAtIndex(index,val)`：在链表中的第`index`个节点之前添加值为`val`的节点。如果`index`等于链表的长度，则该节点将附加到链表的末尾。如果`index`大于链表长度，则不会插入节点。如果`index`小于`0`，则在头部插入节点。
> - `deleteAtIndex(index)`：如果索引`index`有效，则删除链表中的第`index`个节点。

|                                                                                                                                                                                                                                                                                             |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| ```
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
31
32
33
34
35
36
37
38
39
40
41
42
43
44
45
46
47
48
49
50
51
52
53
54
55
56
57
58
59
60
61
62
63
64
65
66
67
68
69
70
71
72
73
74
75
76
77
78
79
80
81
82
83
84
85
86
87
88
89
90
91
92
93
94
95
``` | ```

class MyLinkedList
{
private:
    ListNode* head_;
    int size_;

public:
    struct ListNode
    {
        int val;
        ListNode* next;
        ListNode(int val_, ListNode* next_ = nullptr) : val(val_), next(next_) {}
    };

    MyLinkedList() : head_(nullptr), size_(0) {}

    int get(int index)
    {
        if(index < 0 || index >= size_)
            return -1;

        ListNode* p = head_;
        while(index--) p = p->next;

        return p->val;
    }

    void addAtHead(int val)
    {
        head_ = new ListNode(val, head_);
        size_++;
    }

    void addAtTail(int val)
    {
        if(size_ == 0) addAtHead(val);

        ListNode* p = head_;
        int size = size_;
        while(--size) p = p->next;
        p->next = new ListNode(val);
        size_++;
    }

    void addAtIndex(int index, int val)
    {
        if(index < 0 || index > size_)
            return;
        if(index == size_)
        {
            addAtTail(val);
            return;
        }
        if(index == 0)
        {
            addAtHead(val);
            return;
        }
        ListNode* p = head_;
        ListNode* q = p;
        while(index--)
        {
            q = p;
            p = p->next;
        }

        q->next = new ListNode(val, p);
        size_++;
    }

    void deleteAtIndex(int index)
    {
        if(index < 0 || index >= size_)
            return;

        if(index == 0)
        {
            head_ = head_->next;
            size_--;
            return;
        }

        ListNode* p = head_;
        ListNode* q = p;
        while(index-- && p->next)
        {
            q = p;
            p = p->next;
        }

        q->next = p->next;
        size_--;
    }
};
``` |

##### [](#380-Insert-Delete-GetRandom-O-1-和381-Insert-Delete-GetRandom-O-1-Duplicates-allowed "380. Insert Delete GetRandom O(1)和381. Insert Delete GetRandom O(1) - Duplicates allowed")[380. Insert Delete GetRandom O(1)](https://leetcode.com/problems/insert-delete-getrandom-o1/)和[381. Insert Delete GetRandom O(1) - Duplicates allowed](https://leetcode.com/problems/insert-delete-getrandom-o1-duplicates-allowed/)

> ![title](title.jpg)

|                                                                                                                                                |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| ---------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ```
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
31
32
33
34
35
36
37
38
39
40
41
42
43
44
45
46
47
48
``` | ```

class RandomizedSet
{
private:
    vector<int> nums;
    unordered_map<int, int> mapping;

public:

    RandomizedSet() {}


    bool insert(int val)
    {
        if(mapping.count(val) > 0) return false;
        mapping[val] = nums.size();
        nums.push_back(val);
        return true;
    }


    bool remove(int val)
    {
        if(mapping.count(val) == 0) return false;
        int i = mapping[val];



        mapping[nums.back()] = i;
        mapping.erase(val);

        swap(nums[i], nums.back());
        nums.pop_back();
        return true;
    }


    int getRandom()
    {
        int i = rand() % nums.size();
        return nums[i];
    }
};
``` |

|                                                                                                                                                                                                               |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ```
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
31
32
33
34
35
36
37
38
39
40
41
42
43
44
45
46
47
48
49
50
51
52
53
54
55
56
57
58
59
60
61
62
63
64
65
66
67
68
69
``` | ```

class RandomizedCollection
{
private:
    vector<int> nums;
    unordered_map<int, unordered_set<int>> mapping;

public:

    RandomizedCollection() {}


    bool insert(int val)
    {
        bool res = true;
        if(mapping.count(val) > 0) res = false;

        mapping[val].insert(nums.size());
        nums.push_back(val);
        return res;
    }


    bool remove(int val)
    {
        if(mapping.count(val) == 0) return false;
        int i = *(mapping[val].begin());
        if(val == nums.back())
        {




            mapping[nums.back()].erase(nums.size() - 1);
        }
        else
        {








        	mapping[nums.back()].erase(nums.size() - 1);
        	mapping[nums.back()].insert(i);

            mapping[val].erase(i);
        }

        if(mapping[val].empty()) mapping.erase(val);

        swap(nums[i], nums.back());
        nums.pop_back();
        return true;
    }


    int getRandom()
    {
        int i = rand() % nums.size();
        return nums[i];
    }
};
``` |

##### [](#1114-按序打印 "1114. 按序打印")[1114. 按序打印](https://leetcode-cn.com/problems/print-in-order/)

> 三个不同的线程`A`、`B`、`C`将会共用一个`Foo`实例。
>
> - 一个将会调用`first()`方法
>
> - 一个将会调用`second()`方法
>
> - 还有一个将会调用`third()`方法
>
>   请设计修改程序，以确保`second()`方法在`first()`方法之后被执行，`third()`方法在`second()`方法之后被执行。

|                                                                                                                                                |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| ---------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| ```
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
31
32
33
34
35
36
37
38
39
40
41
42
43
44
45
46
47
48
``` | ```
#include <mutex>
#include <condition_variable>
#include <functional>
#include <thread>
#include <iostream>
using namespace std;

class Foo {
private:
    int counter;
    mutex _mutex;
    condition_variable _cond2b;
    condition_variable _cond2c;

public:
    Foo() : counter(1), _mutex(), _cond2b(), _cond2c() {}

    void first() {
        unique_lock<mutex> lock(_mutex);
        cout << "first" << endl;
        counter = 2;
        _cond2b.notify_one();
    }

    void second() {
        unique_lock<mutex> lock(_mutex);
        _cond2b.wait(lock, [this] { return this->counter == 2; });
        cout << "second" << endl;
        counter = 3;
        _cond2c.notify_one();
    }

    void third() {
        unique_lock<mutex> lock(_mutex);
        _cond2c.wait(lock, [this] { return this->counter == 3; });
        cout << "third" << endl;
    }
};

int main() {
    Foo foo;
    thread A(bind(Foo::first, &foo));
    thread B(bind(Foo::second, &foo));
    thread C(bind(Foo::third, &foo));
    A.join();
    B.join();
    C.join();
}
``` |

##### [](#1115-交替打印FooBar "1115. 交替打印FooBar")[1115. 交替打印FooBar](https://leetcode-cn.com/problems/print-foobar-alternately/)

> 两个不同的线程将会共用一个`FooBar`实例。其中一个线程将会调用`foo()`方法，另一个线程将会调用`bar()`方法。
>
> 请设计修改程序，以确保`"foobar"`被输出`n`次。

|                                                                                                                                                |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| ---------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ```
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
31
32
33
34
35
36
37
38
39
40
41
42
43
44
45
46
47
48
``` | ```

#include <mutex>
#include <condition_variable>
#include <functional>
#include <thread>
#include <iostream>
using namespace std;

class FooBar {
private:
    int n;
    bool counter;
    mutex _mutex;
    condition_variable _cond;

public:
    FooBar(int n) : counter(true) {
        this->n = n;
    }

    void foo() {
        for(int i = 0; i < n; i++) {
            unique_lock<mutex> lock(_mutex);
            _cond.wait(lock, [this] { return this->counter; });
            cout << "foo" << endl;
            counter = false;
            _cond.notify_one();
        }
    }

    void bar() {
        for(int i = 0; i < n; i++) {
            unique_lock<mutex> lock(_mutex);
            _cond.wait(lock, [this] { return !this->counter; });
            cout << "bar" << endl;
            counter = true;
            _cond.notify_one();
        }
    }
};

int main() {
    FooBar foobar(3);
    thread foo(bind(FooBar::foo, &foobar));
    thread bar(bind(FooBar::bar, &foobar));
    foo.join();
    bar.join();
}
``` |

##### [](#1116-打印零与奇偶数 "1116. 打印零与奇偶数")[1116. 打印零与奇偶数](https://leetcode-cn.com/problems/print-zero-even-odd/)

> 相同的一个`ZeroEvenOdd`类实例将会传递给三个不同的线程：
>
> - 线程`A`将调用`zero()`，它只输出`0`。
>
> - 线程`B`将调用`even()`，它只输出偶数。
>
> - 线程`C`将调用`odd(`)，它只输出奇数。
>
>   每个线程都有一个`printNumber`方法来输出一个整数。请修改给出的代码以输出整数序列 `010203040506...`，其中序列的长度必须为`2n`。

|                                                                                                                                                               |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ```
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
31
32
33
34
35
36
37
38
39
40
41
42
43
44
45
46
47
48
49
50
51
52
53
``` | ```
class ZeroEvenOdd {
private:
    int n;
    mutex lockZero;
    mutex lockOdd;
    mutex lockEven;

public:
    ZeroEvenOdd(int n) {
        this->n = n;
        lockOdd.lock();
        lockEven.lock();
    }

    void zero() {
        for(int i = 1; i <= n; i++) {
            lockZero.lock();
            cout << 0 << endl;
            if (i & 1) {
                lockOdd.unlock();
            }
            else {
                lockEven.unlock();
            }
        }
    }

    void even() {
        for(int i = 2; i <= n; i += 2) {
            lockEven.lock();
            cout << i << endl;
            lockZero.unlock();
        }
    }

    void odd() {
        for(int i = 1; i <= n; i += 2) {
            lockOdd.lock();
            cout << i << endl;
            lockZero.unlock();
        }
    }
};

int main() {
    ZeroEvenOdd zero(4);
    thread t1(bind(ZeroEvenOdd::zero, &zero));
    thread t2(bind(ZeroEvenOdd::even, &zero));
    thread t3(bind(ZeroEvenOdd::odd, &zero));
    t1.join();
    t2.join();
    t3.join();
}
``` |

##### [](#1188-设计有限阻塞队列 "1188. 设计有限阻塞队列")[1188. 设计有限阻塞队列](https://leetcode-cn.com/problems/design-bounded-blocking-queue/)

> 实现一个拥有如下方法的线程安全有限阻塞队列：
>
> - `BoundedBlockingQueue(int capacity)`构造方法初始化队列，其中`capacity`代表队列长度上限。
>
> - `void enqueue(int element)`在队首增加一个`element`。如果队列满，调用线程被阻塞直到队列非满。
>
> - `int dequeue()`返回队尾元素并从队列中将其删除。如果队列为空，调用线程被阻塞直到队列非空。
>
> - `int size()`返回当前队列元素个数。
>
>   你的实现将会被多线程同时访问进行测试。每一个线程要么是一个只调用`enqueue`方法的生产者线程，要么是一个只调用`dequeue`方法的消费者线程。`size`方法将会在每一个测试用例之后进行调用。

|                                                                                                               |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| ------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ```
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
31
32
33
34
35
36
37
``` | ```

class BoundedBlockingQueue {
private:
    int _capacity;
    queue<int> _data;
    mutex _mutex;
    condition_variable _notEmpty;
    condition_variable _notFull;

public:
    BoundedBlockingQueue(int capacity) : _capacity(capacity) {}

    void enqueue(int element) {
        unique_lock<mutex> lock(_mutex);
        while(_data.size() >= _capacity) {
            _notFull.wait(lock);
        }
        _data.push(element);
        _notEmpty.notify_one();
    }

    int dequeue() {
        unique_lock<mutex> lock(_mutex);
        while(_data.size() <= 0) {
            _notEmpty.wait(lock);
        }
        int res = _data.front();
        _data.pop();
        _notFull.notify_one();
        return res;
    }

    int size() {
        unique_lock<mutex> lock(_mutex);
        return _data.size();
    }
};
``` |

|                                                                                                            |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| ---------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| ```
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
31
32
33
34
35
36
``` | ```

class BoundedBlockingQueue {
private:
    int _capacity;
    queue<int> _data;
    mutex _mutex;
    condition_variable _cond;

public:
    BoundedBlockingQueue(int capacity) : _capacity(capacity) {}

    void enqueue(int element) {
        unique_lock<mutex> lock(_mutex);
        while(_data.size() >= _capacity) {
            _cond.wait(lock);
        }
        _data.push(element);
        _cond.notify_all();
    }

    int dequeue() {
        unique_lock<mutex> lock(_mutex);
        while(_data.size() <= 0) {
            _cond.wait(lock);
        }
        int res = _data.front();
        _data.pop();
        _cond.notify_all();
        return res;
    }

    int size() {
        unique_lock<mutex> lock(_mutex);
        return _data.size();
    }
};
``` |

##### [](#1226-哲学家进餐 "1226. 哲学家进餐")[1226. 哲学家进餐](https://leetcode-cn.com/problems/the-dining-philosophers/)

> `5`个沉默寡言的哲学家围坐在圆桌前，每人面前一盘意面。叉子放在哲学家之间的桌面上。（`5` 个哲学家，`5`根叉子）
>
> 所有的哲学家都只会在思考和进餐两种行为间交替。哲学家只有同时拿到左边和右边的叉子才能吃到面，而同一根叉子在同一时间只能被一个哲学家使用。每个哲学家吃完面后都需要把叉子放回桌面以供其他哲学家吃面。只要条件允许，哲学家可以拿起左边或者右边的叉子，但在没有同时拿到左右叉子时不能进食。
>
> 假设面的数量没有限制，哲学家也能随便吃，不需要考虑吃不吃得下。
>
> 设计一个进餐规则（并行算法）使得每个哲学家都不会挨饿；也就是说，在没有人知道别人什么时候想吃东西或思考的情况下，每个哲学家都可以在吃饭和思考之间一直交替下去。
>
> ![an\_illustration\_of\_the\_dining\_philosophers\_problem](an_illustration_of_the_dining_philosophers_problem.png)
>
> 哲学家从`0`到`4`按顺时针编号。请实现函数`void wantsToEat(philosopher, pickLeftFork, pickRightFork, eat, putLeftFork, putRightFork)`：
>
> - `philosopher`哲学家的编号。
>
> - `pickLeftFork`和`pickRightFork`表示拿起左边或右边的叉子。
>
> - `eat`表示吃面。
>
> - `putLeftFork`和`putRightFork`表示放下左边或右边的叉子。
>
> - 由于哲学家不是在吃面就是在想着啥时候吃面，所以思考这个方法没有对应的回调。
>
>   给你`5`个线程，每个都代表一个哲学家，请你使用类的同一个对象来模拟这个过程。在最后一次调用结束之前，可能会为同一个哲学家多次调用该函数。

|                                                                                                            |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| ---------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ```
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
31
32
33
34
35
36
``` | ```

class DiningPhilosophers {
private:
    array<mutex, 5> mutexs;

public:
    DiningPhilosophers() {}

    void wantsToEat(int philosopher) {
        int left = philosopher;
        int right = (philosopher + 1) % 5;
        lock(mutexs[left], mutexs[right]);

        {
            lock_guard<mutex> lock_left(mutexs[left], adopt_lock);
            lock_guard<mutex> lock_right(mutexs[right], adopt_lock);
            cout << "ID: " << philosopher << " eatting" << endl;
        }
    }
};

int main() {
    DiningPhilosophers dining;
    thread t1(bind(DiningPhilosophers::wantsToEat, &dining, placeholders::_1), 0);
    thread t2(bind(DiningPhilosophers::wantsToEat, &dining, placeholders::_1), 1);
    thread t3(bind(DiningPhilosophers::wantsToEat, &dining, placeholders::_1), 2);
    thread t4(bind(DiningPhilosophers::wantsToEat, &dining, placeholders::_1), 3);
    thread t5(bind(DiningPhilosophers::wantsToEat, &dining, placeholders::_1), 4);
    t1.join();
    t2.join();
    t3.join();
    t4.join();
    t5.join();
}
``` |

|                                                                                                         |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| ------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ```
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
31
32
33
34
35
``` | ```

class DiningPhilosophers {
private:
    array<mutex, 5> mutexs;
    mutex door;

public:
    DiningPhilosophers() {}

    void wantsToEat(int philosopher) {
        int left = philosopher;
        int right = (philosopher + 1) % 5;

        door.lock();
        lock_guard<mutex> lock_left(mutexs[left]);
        lock_guard<mutex> lock_right(mutexs[right]);
        door.unlock();

        cout << "ID " << philosopher << " is eatting" << endl;
    }
};

int main() {
    DiningPhilosophers dining;
    thread t1(bind(DiningPhilosophers::wantsToEat, &dining, placeholders::_1), 0);
    thread t2(bind(DiningPhilosophers::wantsToEat, &dining, placeholders::_1), 1);
    thread t3(bind(DiningPhilosophers::wantsToEat, &dining, placeholders::_1), 2);
    thread t4(bind(DiningPhilosophers::wantsToEat, &dining, placeholders::_1), 3);
    thread t5(bind(DiningPhilosophers::wantsToEat, &dining, placeholders::_1), 4);
    t1.join();
    t2.join();
    t3.join();
    t4.join();
    t5.join();
}
``` |

`Undetermined`

##### [](#343-Integer-Break "343. Integer Break")[343. Integer Break](https://leetcode.com/problems/integer-break/)

> 给定一个整数`n`，将其分解为`k`个正整数之和，其中`k >= 2`，并使这些整数的乘积最大化。返回你可以获得的最大乘积。
>
> 说明: 你可以假设`n`不小于`2`且不大于`58`。

|                                                            |                                                                                                                                                                                                                                                                                                                                                                                     |
| ---------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ```
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
``` | ```

int integerBreak(int n) {
    if(n <= 3) {
        return n - 1;
    }
    vector<int> dp(n + 1);
    dp[2] = 1;
    dp[3] = 2;
    for(int i = 4; i <= n; i++) {

        for(int j = 2; j <= i - 2; j++) {




            dp[i] = max({dp[i], j * (i - j), j * dp[i - j]});
        }
    }
    return dp[n];
}
``` |

|                                                |                                                                                                                                                                                                                                                                |
| ---------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ```
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
``` | ```

int integerBreak(int n) {
    if(n <= 3) {
        return n - 1;
    }
    int a = n / 3, b = n % 3;
    if(b == 0) {
        return pow(3, a);
    }
    else if(b == 1) {
        return 4 * pow(3, a - 1);
    }
    return 2 * pow(3, a - 1);
}
``` |

##### [](#509-Fibonacci-Number "509. Fibonacci Number")[509. Fibonacci Number](https://leetcode.com/problems/fibonacci-number/)

> 斐波那契数，通常用`F(n)`表示，形成的序列称为斐波那契数列 。该数列由`0`和`1`开始，后面的每一项数字都是前面两项数字的和。也就是：
>
> `F(0) = 0，F(1) = 1`\
> `F(n) = F(n - 1) + F(n - 2)`，其中`n > 1`
>
> 给你`n`，请计算`F(n)`。

|                                    |                                                                                                                                                                                                                     |
| ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ```
1
2
3
4
5
6
7
8
9
10
11
12
``` | ```

int fib(int n) {
    int dp_0 = 0;
    int dp_1 = 1;
    int mod = 1e9 + 7;
    while(n-- > 0) {
        int temp = dp_0 + dp_1;
        dp_0 = dp_1;
        dp_1 = temp % mod;
    }
    return dp_0;
}
``` |

|                                          |                                                                                                                                                                                                                                                                                                                                  |
| ---------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ```
1
2
3
4
5
6
7
8
9
10
11
12
13
14
``` | ```

int fib(int n) {
    vector<int> memo(n + 1);
    auto recur = [&](auto&& recur, int n) {
        if(n < 2) {
            return n;
        }
        if(memo[n] != 0) {
            return memo[n];
        }
        return memo[n] = recur(recur, n - 1) + recur(recur, n - 2);
    };
    return recur(recur, n);
}
``` |

[从小白到大神都会遇到的经典面试题 —— 斐波那契数列\_0 error(s)-CSDN博客](https://blog.csdn.net/shenmingxueIT/article/details/117332922?spm=1001.2014.3001.5501)

|           |         |
| --------- | ------- |
| ```
1
``` | ```
``` |

##### [](#50-Pow-x-n "50. Pow(x, n)")[50. Pow(x, n)](https://leetcode.com/problems/powx-n/)

> 实现`pow(x, n)`，即计算`x`的`n`次幂函数（即 x^n^ ）。

|                                                                                             |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| ------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ```
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
31
``` | ```

double myPow(double x, long n)
{
    if(n == 0) {
        return 1;
    }
    if(n < 0) {
        return myPow(1.0 / x, -n);
    }
    if(n % 2 == 1) {
        return x * myPow(x, n - 1);
    }
    return myPow(x * x, n / 2);
}

double myPow(double x, long n) {
    if(n < 0) {
        x = 1.0 / x;
        n = -n;
    }
    double res = 1;
    while(n != 0) {
        if(n & 1) {
            res *= x;
        }
        n >>= 1;
        x *= x;
    }
    return res;
}
``` |

##### [](#372-Super-Pow "372. Super Pow")[372. Super Pow](https://leetcode.com/problems/super-pow/)

> 你的任务是计算 a^b^ 对`1337`取模，`a`是一个正整数，`b`是一个非常大的正整数且会以数组形式给出。
>
> ![img](formu1.png)
>
> **Constraints:**
>
> - `1 <= a <= 231 - 1`
> - `1 <= b.length <= 2000`
> - `0 <= b[i] <= 9`
> - `b` doesn’t contain leading zeros.
>
> **Example 1:**
>
> |             |                                              |
> | ----------- | -------------------------------------------- |
> | ```
> 1
> 2
> ``` | ```
> Input: a = 2, b = [1,0]
> Output: 1024
> ``` |
>
> **Example 2:**
>
> |             |                                                   |
> | ----------- | ------------------------------------------------- |
> | ```
> 1
> 2
> ``` | ```
> Input: a = 1, b = [4,3,3,8,5,2]
> Output: 1
> ``` |
>
> **Example 3:**
>
> |             |                                                         |
> | ----------- | ------------------------------------------------------- |
> | ```
> 1
> 2
> ``` | ```
> Input: a = 2147483647, b = [2,0,0]
> Output: 1198
> ``` |

|                                                                                                                     |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| ------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ```
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
31
32
33
34
35
36
37
38
39
``` | ```

int superPow(int a, vector<int>& b)
{
    if(b.empty()) return 1;
    int back = b.back();
    b.pop_back();
    return mypow(a, back) * mypow(superPow(a, b), 10) % 1337;
}

int mypow(int a, int n)
{
    int res = 1;


    a %= 1337;
    while(n-- > 0)
    {

        res *= a;

        res %= 1337;
    }
    return res;
}

int mypow(int a, int n)
{
    if(n == 0) return 1;
    a %= 1337;
    if(n % 2 == 1)
        return a * mypow(a, n - 1) % 1337;
    return mypow(a * a % 1337, n / 2);
}
``` |

##### [](#779-K-th-Symbol-in-Grammar "779. K-th Symbol in Grammar")[779. K-th Symbol in Grammar](https://leetcode.com/problems/k-th-symbol-in-grammar/)

> 在第一行我们写上一个`0`。接下来的每一行，将前一行中的`0`替换为`01`，`1`替换为`10`。
>
> 给定行数`N`和序数`K`，返回第`N`行中第`K`个字符。
>
> **Note:**
>
> 1. `N` will be an integer in the range `[1, 30]`.
>
> 2. `K` will be an integer in the range `[1, 2^(N-1)]`.
>
>    |                                                      |                                                                                                                                                                                                           |
>    | ---------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
>    | ```
>    1
>    2
>    3
>    4
>    5
>    6
>    7
>    8
>    9
>    10
>    11
>    12
>    13
>    14
>    15
>    16
>    17
>    18
>    ``` | ```
>    Examples:
>    Input: N = 1, K = 1
>    Output: 0
>
>    Input: N = 2, K = 1
>    Output: 0
>
>    Input: N = 2, K = 2
>    Output: 1
>
>    Input: N = 4, K = 5
>    Output: 1
>
>    Explanation:
>    row 1: 0
>    row 2: 01
>    row 3: 0110
>    row 4: 01101001
>    ``` |

|                       |                                                                                                                                                                                  |
| --------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ```
1
2
3
4
5
6
7
``` | ```
int kthGrammar(int N, int K)
{
    if(K == 1) return 0;
    int n = pow(2, N - 2);
    if(K > n) return 1 - kthGrammar(N - 1, K - n);
    return kthGrammar(N - 1, K);
}
``` |
````

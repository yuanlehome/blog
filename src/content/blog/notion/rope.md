---
title: RoPE 究竟是怎么计算的
slug: rope
date: '2026-01-04'
tags: []
status: published
cover: ''
lastEditedTime: '2026-01-04T14:28:00.000Z'
updated: '2026-01-04T14:28:00.000Z'
source: notion
notion:
  id: 2d322dca-4210-8074-95ce-ec86131a7787
---

---

## 一、RoPE 到底改了什么？

Attention 里我们原来用：

- $Q, K, V$ 计算 $\text{Attn}(Q, K, V)$

RoPE 做的是把 **Q/K 先旋转**：

- $Q' = \text{RoPE}(Q, \text{pos})$
- $K' = \text{RoPE}(K, \text{pos})$
- 然后用 $Q', K', V$ 做 attention

---

## 二、数学形式：2D 旋转（每两维一组）

对每个 position $p$，对某一对维度 $(u,v)$ 做旋转：

\$\begin{bmatrix}
x'\_u \\
x'\_v
\end{bmatrix}
=============

\begin{bmatrix}
\cos\theta & -\sin\theta \\
\sin\theta & \cos\theta
\end{bmatrix}
\begin{bmatrix}
x_u \\
x_v
\end{bmatrix}\$

工程实现常写成统一模板：

$x_{\text{rot}} = x \cdot \cos + \text{rotate\_half}(x)\cdot \sin$

关键：

- **`rotate_half`** **定义了“哪两维是一对”**
- **`cos/sin`** **展开方式必须和这个配对一致**

---

## 三、两种 RoPE style：差别只在“怎么配对维度”

假设 head_dim $D=8$，向量：$x=[x_0,x_1,x_2,x_3,x_4,x_5,x_6,x_7]$

### 3.1 Adjacent-pair（相邻配对 / even-odd）

配对方式：

- $(0,1), (2,3), (4,5), (6,7)$

对应的 `rotate_half`（相邻两两转）会把每对做 $(−y,x)$：

- 输出：`[-x1, x0, -x3, x2, -x5, x4, -x7, x6]`

### 3.2 NeoX-style（前后半配对 / split-half）

配对方式：

- $(0,4),(1,5),(2,6),(3,7)$

对应的 `rotate_half`（前后半互换再取负）：

- 输出：`[-x4, -x5, -x6, -x7, x0, x1, x2, x3]`

---

## 四、最容易踩坑的点：cos/sin 怎么“铺满到 D 维”

先算基础的 $\cos,\sin$（每对一个频率），它们天然是长度 $D/2$ 的向量：

- `cos_base = [c0 c1 c2 c3]`
- `sin_base = [s0 s1 s2 s3]`

### 4.1 NeoX-style 的展开（**拼两份**）

因为配对是 `(i, i+D/2)`，所以要变成：

- `cos_full = [c0 c1 c2 c3 c0 c1 c2 c3]`
- `sin_full = [s0 s1 s2 s3 s0 s1 s2 s3]`

### 4.2 Adjacent-pair 的展开（**每个重复两次**）

因为配对是 `(2i, 2i+1)`，所以要变成：

- `cos_full = [c0 c0 c1 c1 c2 c2 c3 c3]`
- `sin_full = [s0 s0 s1 s1 s2 s2 s3 s3]`

**结论（牢记）**：

- NeoX-style：`rotate_half = cat([-x2, x1])` 必须配 `cos/sin = cat([cos, cos])`
- Adjacent-pair：`rotate_half` 用 `::2 / 1::2` 必须配 `repeat_interleave(2)`

不匹配通常会出现：

> 不报 shape 错，但模型效果/困惑度明显崩掉

---

## 五、参考实现（布局：`[B, T, H, D]`）

### 5.1 两种 rotate_half

```python
import torch

def rotate_half_adjacent(x):  # x: [..., rotary_dim]
    # pair: (0,1), (2,3), ...
    x_even = x[..., ::2]
    x_odd  = x[..., 1::2]
    out = torch.stack((-x_odd, x_even), dim=-1)
    return out.flatten(-2)  # restore last dim

def rotate_half_neox(x):  # x: [..., rotary_dim]
    # pair: (0, D/2), (1, D/2+1), ...
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((-x2, x1), dim=-1)
```

### 5.2 构建 cos/sin cache（核心：不同 style 的展开）

```python
def build_rope_cache(seq_len, rotary_dim, base=10000, device=None,
                     dtype=torch.float16, style="neox"):
    assert rotary_dim % 2 == 0
    device = device or "cuda"

    # inv_freq: [rotary_dim/2]
    i = torch.arange(0, rotary_dim, 2, device=device, dtype=torch.float32)
    inv_freq = 1.0 / (base ** (i / rotary_dim))

    # t: [seq_len]
    t = torch.arange(seq_len, device=device, dtype=torch.float32)

    # freqs: [seq_len, rotary_dim/2]
    freqs = torch.einsum("t,f->tf", t, inv_freq)

    cos = freqs.cos().to(dtype)  # [T, D/2]
    sin = freqs.sin().to(dtype)  # [T, D/2]

    # expand to [T, D]
    if style == "neox":
        cos = torch.cat([cos, cos], dim=-1)  # [c.., c..]
        sin = torch.cat([sin, sin], dim=-1)
    elif style == "adjacent":
        cos = cos.repeat_interleave(2, dim=-1)  # [c0,c0,c1,c1,...]
        sin = sin.repeat_interleave(2, dim=-1)
    else:
        raise ValueError(f"unknown style={style}")

    # broadcast shape for q/k: [B,T,H,rotary_dim]
    cos = cos[:, None, None, :]  # [T,1,1,D]
    sin = sin[:, None, None, :]  # [T,1,1,D]
    return cos, sin
```

### 5.3 应用到 Q/K（只旋转前 rotary_dim）

```python
def apply_rope(q, k, cos, sin, rotary_dim, style="neox"):
    # q,k: [B, T, H, D]
    q1, q2 = q[..., :rotary_dim], q[..., rotary_dim:]
    k1, k2 = k[..., :rotary_dim], k[..., rotary_dim:]

    rot = rotate_half_neox if style == "neox" else rotate_half_adjacent

    q1 = q1 * cos + rot(q1) * sin
    k1 = k1 * cos + rot(k1) * sin

    q = torch.cat([q1, q2], dim=-1)
    k = torch.cat([k1, k2], dim=-1)
    return q, k
```

## 六、设一个最小例子：D = 8（NeoX style）

```plain text
x = [x0, x1, x2, x3, x4, x5, x6, x7]   # 一条 head 的某个 token 的向量
```

`rotate_half(x)` 做什么？

```python
x1 = x[..., :4]   = [x0, x1, x2, x3]
x2 = x[...,4:]   = [x4, x5, x6, x7]

rotate_half(x) = [-x4, -x5, -x6, -x7, x0, x1, x2, x3]
```

然后 RoPE：

```python
q_rot = x * cos + rotate_half(x) * sin
```

也就是对每个维度 $i$ 都做：

```plain text
q_rot[i] = x[i] * cos[i] + rotate_half(x)[i] * sin[i]
```

用我们上面的 $cos/sin$ 记号 (`cos = [c0..c3,c0..c3]`, `sin=[s0..s3,s0..s3]`)，展开就是：

```plain text
q0 = x0 * c0 + (-x4) * s0 = x0 * c0 - x4 * s0
q1 = x1 * c1 + (-x5) * s1 = x1 * c1 - x5 * s1
q2 = x2 * c2 + (-x6) * s2 = x2 * c2 - x6 * s2
q3 = x3 * c3 + (-x7) * s3 = x3 * c3 - x7 * s3

q4 = x4 * c0 + ( x0) * s0 = x4 * c0 + x0 * s0
q5 = x5 * c1 + ( x1) * s1 = x5 * c1 + x1 * s1
q6 = x6 * c2 + ( x2) * s2 = x6 * c2 + x2 * s2
q7 = x7 * c3 + ( x3) * s3 = x7 * c3 + x3 * s3
```

把结果按 pair 重组一下就非常清晰了。使用频率`(c0,s0)`的是`(x0,x4)`这一对：

```plain text
q0 =  x0 * c0 - x4 * s0
q4 =  x4 * c0 + x0 * s0
```

这是标准二维旋转：

# \$\begin{bmatrix}q0\q4\end{bmatrix}

\begin{bmatrix}
\cos\theta_0 & -\sin\theta_0 \\
\sin\theta_0 & \cos\theta_0
\end{bmatrix}
\begin{bmatrix}x0\x4\end{bmatrix}\$

其他同理：

使用`(c1,s1)`的是`(x1,x5)`：

```plain text
q1 =  x1 * c1 - x5 * s1
q5 =  x5 * c1 + x1 * s1
```

使用`(c2,s2)`的是`(x2,x6)`：

```plain text
q2 =  x2 * c2 - x6 * s2
q6 =  x6 * c2 + x2 * s2
```

使用`(c3,s3)`的是`(x3,x7)`：

```plain text
q3 =  x3 * c3 - x7 * s3
q7 =  x7 * c3 + x3 * s3
```

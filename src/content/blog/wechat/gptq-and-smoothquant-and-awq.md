---
title: GPTQ & SmoothQuant & AWQ 代码解析
slug: gptq-and-smoothquant-and-awq
date: '2026-01-31'
tags: []
status: published
source_url: 'https://mp.weixin.qq.com/s/34a7elkMSDdFVSLEdO7MWg'
source_author: GiantPandaLLM
imported_at: '2026-01-31T05:43:02.534Z'
source:
  title: mp.weixin.qq.com
  url: 'https://mp.weixin.qq.com/s/34a7elkMSDdFVSLEdO7MWg'
cover: /images/wechat/gptq-and-smoothquant-and-awq/001-a2d0e0ed.gif
---

![图片](/images/wechat/gptq-and-smoothquant-and-awq/001-a2d0e0ed.gif)

# GPTQ & SmoothQuant & AWQ 代码解析

本文主要是对 LLM PTQ 量化方向的几个经典算法（GPTQ、SmoothQuant、AWQ）的代码实现进行介绍，一方面是为了加深对算法的理解，另一方面也是想看看有什么值得借鉴的地方。

## 一、GPTQ

GPTQ 在 LLM 量化 W4A16 方向的地位毋庸置疑，它的出发点很朴素，就是试图最小化权重量化后和量化前的层函数误差，对这个最优化问题进行求解后结果包含二阶的 Hessian matrix（海森矩阵），详细数学推理公式见文章 HELLO 七仔：GPTQ 模型量化，论文见：GPTQ，这里不做详细解释。本质上，它的核心流程其实就是量化-补偿-量化-补偿的迭代，下图可以说明这个过程。

![图片](/images/wechat/gptq-and-smoothquant-and-awq/002-09cd47df.jpg)

本文以 GPTQ-for-LLaMa（<https://github.com/qwopqwop200/GPTQ-for-LLaMa>）代码仓库为例来讲解 GPTQ 算法的实现，这个仓库主要是在 LlaMa 模型上应用 GPTQ 算法实现权重的 4 bit 量化。先来看下 Llama 中 DeocoderLayer 的基本结构，主要是由 LlamaAttention、LlamaMLP 和两个 LlamaRMSNorm 构成，GPTQ 会对其中 LlamaAttention 和 LlamaMLP 层中的 Linear 层权重进行量化。

```text
LlamaDecoderLayer(
  (self_attn): LlamaAttention(
    (q_proj): Linear(in_features=4096, out_features=4096, bias=False)
    (k_proj): Linear(in_features=4096, out_features=1024, bias=False)
    (v_proj): Linear(in_features=4096, out_features=1024, bias=False)
    (o_proj): Linear(in_features=4096, out_features=4096, bias=False)
    (rotary_emb): LlamaRotaryEmbedding()
  )
  (mlp): LlamaMLP(
    (gate_proj): Linear(in_features=4096, out_features=14336, bias=False)
    (up_proj): Linear(in_features=4096, out_features=14336, bias=False)
    (down_proj): Linear(in_features=14336, out_features=4096, bias=False)
    (act_fn): SiLU()
  )
  (input_layernorm): LlamaRMSNorm()
  (post_attention_layernorm): LlamaRMSNorm()
)
```

整体量化过程大致可以分为 3 个部分：

1. 利用 calibration data 计算 Hessian 矩阵，对模型进行逐层 weight 量化。
2. 保存量化后的 weight。
3. 代码主要在 llama.py、gptq.py、quantizer.py 和 quant_linear.py 几个文件，由于篇幅有限我们仅关注核心代码部分。

### 1. 计算 Hessian 矩阵

Hessian 矩阵会用于后面逐层量化过程中的损失和补偿计算，所以需要先离线计算得到。实现方式是在初始化 GPTQ 后在每一层注册 hook，通过 hook 的方式在 layer forward 后使用 calibration data 的 input 来生成 Hessian 矩阵，这种计算方式还挺常见的，后面的算法中也有用到。下面这段代码即添加 hook 函数来利用 calibration data 进行计算，计算 Hessian 矩阵的逻辑体现在 `add_batch` 函数中。

```python
            for name in subset:
                gptq[name] = GPTQ(subset[name], observe=args.observe)
                gptq[name].quantizer.configure(args.wbits, perchannel=True, sym=args.sym, mse=False)

            # generate Hessian H by calibration data
            def add_batch(name):

                def tmp(_, inp, out):
                    gptq[name].add_batch(inp[0].data, out.data)

                return tmp

            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            for j in range(args.nsamples):
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
            for h in handles:
                h.remove()
```

为了利用所有的校准数据，这里通过迭代的方式将每组数据计算的 Hessian 矩阵值进行求和然后取平均，代码实现是迭代逐渐平均叠加的过程。

![图片](/images/wechat/gptq-and-smoothquant-and-awq/003-80a15ba3.jpg)

```python
    def add_batch(self, inp, out):
        # Hessian H = 2 X XT + λ I
        if self.observe:
            self.inp1 = inp
            self.out1 = out
        else:
            self.inp1 = None
            self.out1 = None

        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear) or isinstance(self.layer, transformers.Conv1D):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        if isinstance(self.layer, nn.Conv2d):
            unfold = nn.Unfold(self.layer.kernel_size, dilation=self.layer.dilation, padding=self.layer.padding, stride=self.layer.stride)
            inp = unfold(inp)
            inp = inp.permute([1, 0, 2])
            inp = inp.flatten(1)
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())
```

### 2. 逐层 weight 量化

有了 Hessian 矩阵后，便可以用来计算量化误差从而更新权重了，这里是逐层使用 `fasterquant` 方法作为入口来进行量化处理。

```text
            for name in subset:
                scale, zero, g_idx, error = gptq[name].fasterquant(percdamp=args.percdamp, groupsize=args.groupsize, actorder=args.act_order, name=name)
                quantizers['model.layers.%d.%s' % (i, name)] = (gptq[name].quantizer.cpu(), scale.cpu(), zero.cpu(), g_idx.cpu(), args.wbits, args.groupsize)
```

在 `fasterquant` 方法中首先需要根据给定的权重值确定量化所需要的 scale 和 zeropoint，由于采用的 per-channel 量化所以每个 channel 都需要计算它的 scale 和 zeropoint，这里采用的是最简单的 min-max 方法来计算 scale 和 zeropoint，代码如下：

```python
    def find_params(self, x, weight=False):
        dev = x.device
        self.maxq = self.maxq.to(dev)

        shape = x.shape
        if self.perchannel:
            if weight:
                x = x.flatten(1)
            else:
                if len(shape) == 4:
                    x = x.permute([1, 0, 2, 3])
                    x = x.flatten(1)
                if len(shape) == 3:
                    x = x.reshape((-1, shape[-1])).t()
                if len(shape) == 2:
                    x = x.t()
        else:
            x = x.flatten().unsqueeze(0)

        tmp = torch.zeros(x.shape[0], device=dev)
        xmin = torch.minimum(x.min(1)[0], tmp)
        xmax = torch.maximum(x.max(1)[0], tmp)

        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax)
            tmp = xmin < 0
            if torch.any(tmp):
                xmin[tmp] = -xmax[tmp]
        tmp = (xmin == 0) & (xmax == 0)
        xmin[tmp] = -1
        xmax[tmp] = +1

        if self.maxq < 0:
            self.scale = xmax
            self.zero = xmin
        else:
            self.scale = (xmax - xmin) / self.maxq
            if self.sym:
                self.zero = torch.full_like(self.scale, (self.maxq + 1) / 2)
            else:
                self.zero = torch.round(-xmin / self.scale)
        if not self.perchannel:
            if weight:
                tmp = shape[0]
            else:
                tmp = shape[1] if len(shape) != 3 else shape[2]
            self.scale = self.scale.repeat(tmp)
            self.zero = self.zero.repeat(tmp)

        if weight:
            shape = [-1] + [1] * (len(shape) - 1)
            self.scale = self.scale.reshape(shape)
            self.zero = self.zero.reshape(shape)
            return
        if len(shape) == 4:
            self.scale = self.scale.reshape((1, -1, 1, 1))
            self.zero = self.zero.reshape((1, -1, 1, 1))
        if len(shape) == 3:
            self.scale = self.scale.reshape((1, 1, -1))
            self.zero = self.zero.reshape((1, 1, -1))
        if len(shape) == 2:
            self.scale = self.scale.unsqueeze(0)
            self.zero = self.zero.unsqueeze(0)
```

接着为了增强数值稳定性加速收敛，需要完成完整的 Hessian 矩阵计算和 cholesky 分解，过程见代码注解。

```text
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0
        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        #  Hessian H = 2 X XT + λ I
        # 在使用 Hessian 矩阵进行优化时，阻尼（dampening）是一种常见技术，用于改善数值稳定性和收敛性
        H[diag, diag] += damp
        # cholesky分解Hessian 矩阵，增强数值稳定性
        # Cholesky 分解的下三角矩阵
        H = torch.linalg.cholesky(H)
        # Hessian 矩阵的逆
        H = torch.cholesky_inverse(H)
        # 逆矩阵分解的上三角矩阵
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H
```

这样准备工作都完成了就可以进行论文中的算法具体代码实现了，下面这段代码就是完全对应论文中的伪代码实现，值得一提的是这里可以指定 `groupsize` 来对量化的范围进行进一步的缩减，一定程度上可以减少离群值的影响。这里量化的 per-channel scale 和 zero 会随着 $W$ 的迭代更新而发生变化，最终返回 scale, zero, g_idx。

![图片](/images/wechat/gptq-and-smoothquant-and-awq/004-60185fde.jpg)

```text
        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]
                # use groupsize column for quantization
                if groupsize != -1:
                    if (i1 + i) % groupsize == 0:
                        self.quantizer.find_params(W[:, (i1 + i):(i1 + i + groupsize)], weight=True)

                    if ((i1 + i) // groupsize) - now_idx == -1:
                        scale.append(self.quantizer.scale)
                        zero.append(self.quantizer.zero)
                        now_idx += 1

                q = self.quantizer.quantize(w.unsqueeze(1)).flatten()
                Q1[:, i] = q
                Losses1[:, i] = (w - q)**2 / d**2

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            Q[:, i1:i2] = Q1
            Losses[:, i1:i2] = Losses1 / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        torch.cuda.synchronize()
        error = torch.sum(Losses).item()

        groupsize = groupsize if groupsize != -1 else self.columns
        g_idx = [i // groupsize for i in range(self.columns)]
        g_idx = torch.tensor(g_idx, dtype=torch.int32, device=Q.device)
        if actorder:
            invperm = torch.argsort(perm)
            Q = Q[:, invperm]
            g_idx = g_idx[invperm]

        if isinstance(self.layer, transformers.Conv1D):
            Q = Q.t()

        self.print_loss(name=name, q_weight=Q, weight_error=error, timecost=(time.time() - tick))

        if scale == []:
            scale.append(self.quantizer.scale)
            zero.append(self.quantizer.zero)
        scale = torch.cat(scale, dim=1)
        zero = torch.cat(zero, dim=1)
        return scale, zero, g_idx, error
```

其中 quantize 函数最终调用的 \_quantize 实现如下，本质上是伪量化（包含量化和反量化）。

```python
    def _quantize(self, x, scale, zero, maxq):
        if maxq < 0:
            return (x > scale / 2).float() * scale + (x < zero / 2).float() * zero
        q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
        return scale * (q - zero)
```

### 3. 保存量化 weight

之前的步骤中量化和反量化后计算 lose 都是浮点位数的，所以并没有生成 wbit 位 format 的数值内容，在 `llama_pack` 方法中通过 model 和之前得到的 `quantizer`(scale, zero) 来生成 wbit 位数表达格式的量化模型，其定义如下所示

```python
def llama_pack(model, quantizers, wbits, groupsize):
    layers = find_layers(model)
    layers = {n: layers[n] for n in quantizers}
    quant.make_quant_linear(model, quantizers, wbits, groupsize)
    qlayers = find_layers(model, [quant.QuantLinear])
    print('Packing ...')
    for name in qlayers:
        print(name)
        quantizers[name], scale, zero, g_idx, _, _ = quantizers[name]
        qlayers[name].pack(layers[name], scale, zero, g_idx)
    print('Done.')
    return model
```

其中 `quantizers` 来自量化后的返回，它是一个 dict 里面保存了每一个层和它对应的 `quantizer`、`scale`、`zero`、`group_idx` 等信息，其中 `quantizer` 是 layer-level 的，`zero` 和 `scale` 是 group-level 的。

```text
quantizers['model.layers.%d.%s' % (i, name)] = (gptq[name].quantizer.cpu(), scale.cpu(), zero.cpu(), g_idx.cpu(), args.wbits, args.groupsize)
```

接下来逐步介绍 llama_pack 的实现，首先由 `make_quant_linear` 递归地将所有 `Linear` 替换为 `QuantLinear`

```python
def make_quant_linear(module, names, bits, groupsize, name=''):
    if isinstance(module, QuantLinear):
        return
    for attr in dir(module):
        tmp = getattr(module, attr)
        name1 = name + '.' + attr if name != '' else attr
        if name1 in names:
            delattr(module, attr)
            setattr(module, attr, QuantLinear(bits, groupsize, tmp.in_features, tmp.out_features, tmp.bias is not None))
    for name1, child in module.named_children():
        make_quant_linear(child, names, bits, groupsize, name + '.' + name1 if name != '' else name1)
```

其中 `QuantLinear` 的定义如下，通过 `qweight`、`qzeros` 和 `scales`、`g_idx` 等属性来保存量化后的低比特信息。

```python
class QuantLinear(nn.Module):

    def __init__(self, bits, groupsize, infeatures, outfeatures, bias):
        super().__init__()
        if bits not in [2, 4, 8]:
            raise NotImplementedError("Only 2,4,8 bits are supported.")
        self.infeatures = infeatures
        self.outfeatures = outfeatures
        self.bits = bits
        self.maxq = 2**self.bits - 1
        self.groupsize = groupsize if groupsize != -1 else infeatures

        self.register_buffer('qweight', torch.zeros((infeatures // 32 * self.bits, outfeatures), dtype=torch.int32))
        self.register_buffer('qzeros', torch.zeros((math.ceil(infeatures / self.groupsize), outfeatures // 32 * self.bits), dtype=torch.int32))
        self.register_buffer('scales', torch.zeros((math.ceil(infeatures / self.groupsize), outfeatures), dtype=torch.float16))
        self.register_buffer('g_idx', torch.tensor([i // self.groupsize for i in range(infeatures)], dtype=torch.int32))
        if bias:
            self.register_buffer('bias', torch.zeros((outfeatures), dtype=torch.float16))
        else:
            self.bias = None
```

接着对每个 `QuantLinear` 层使用 pack 来重新打包量化后的权重数据。实际的存储数据格式是 uint32，所以针对 4bit 量化值，单个 qweight 可以存储 8 个权重值。

1. 首先对原 weight 利用 scale 和 zero 计算出 int4 范围的 int 权重表示。
2. 再合并成 uint32 格式进行存储，这里采用了 intweight 左移和或运算来完成低比特到 32bit 的转存；zeros 也是类似逻辑转成 qzeros 表示；scales 直接转为 half 格式保存；g_idx 保持不变；这样就完成了对原 weight 的压缩转换。
3. 推理的时候需要利用 scales 和 zeros 进行反量化再进行计算。

这里其实有一点疑惑，就是对权重进行 quant 的过程只用到了之前得到的 per-channel scale 和 zero，没有体现前述逐 block 量化过程中对权重的补偿，因为这里用的 weight 还是原始模型的 weight 并不是第二步量化过程中损失补偿修改后的 weight。

pack 函数实现如下。

```python
    def pack(self, linear, scales, zeros, g_idx=None):
        self.g_idx = g_idx.clone() if g_idx is not None else self.g_idx

        scales = scales.t().contiguous()
        zeros = zeros.t().contiguous()
        scale_zeros = zeros * scales
        self.scales = scales.clone().half()
        if linear.bias is not None:
            self.bias = linear.bias.clone().half()

        intweight = []
        for idx in range(self.infeatures):
            intweight.append(torch.round((linear.weight.data[:, idx] + scale_zeros[self.g_idx[idx]]) / self.scales[self.g_idx[idx]]).to(torch.int)[:, None])
        intweight = torch.cat(intweight, dim=1)
        intweight = intweight.t().contiguous()
        intweight = intweight.numpy().astype(np.uint32)
        qweight = np.zeros((intweight.shape[0] // 32 * self.bits, intweight.shape[1]), dtype=np.uint32)
        i = 0
        row = 0
        while row < qweight.shape[0]:
            if self.bits in [2, 4, 8]:
                for j in range(i, i + (32 // self.bits)):
                    qweight[row] |= intweight[j] << (self.bits * (j - i))
                i += 32 // self.bits
                row += 1
            else:
                raise NotImplementedError("Only 2,4,8 bits are supported.")

        qweight = qweight.astype(np.int32)
        self.qweight = torch.from_numpy(qweight)

        zeros -= 1
        zeros = zeros.numpy().astype(np.uint32)
        qzeros = np.zeros((zeros.shape[0], zeros.shape[1] // 32 * self.bits), dtype=np.uint32)
        i = 0
        col = 0
        while col < qzeros.shape[1]:
            if self.bits in [2, 4, 8]:
                for j in range(i, i + (32 // self.bits)):
                    qzeros[:, col] |= zeros[:, j] << (self.bits * (j - i))
                i += 32 // self.bits
                col += 1
            else:
                raise NotImplementedError("Only 2,4,8 bits are supported.")

        qzeros = qzeros.astype(np.int32)
        self.qzeros = torch.from_numpy(qzeros)
```

实测下来对 Llama2-7b 模型进行 GPTQ 量化在 4090 上耗时 11min 左右，速度还行，最后一层量化误差也还可以。

![图片](/images/wechat/gptq-and-smoothquant-and-awq/005-1d2e72a6.jpg)

## 二、SmoothQuant

SmoothQuant（<https://arxiv.org/abs/2211.10438>）也是应用很广泛的 LLM 量化算法，它对权重和激活值都进行量化，是一个 W8A8 算法。它发现权重比较容易量化，激活值不易量化，因为有离群值，因此提出了在 channel 维度上对激活值和权重进行了平滑处理，这样易于量化的方案。本文针对这个算法基于官方 Repo 进行代码分析。Repo 中给出的 `generate_act_scales.py` 和 `export_int8_model.py` 脚本用于生成一个 INT8 类型的 OPT 模型。整体上它也是分成 3 个步骤：

1. 根据校准数据集生成激活值 scale。
2. 使用激活值 scale smooth 模型。
3. 量化模型。

### 1. 根据校准数据生成激活值 scale

首先使用 `generate_act_scales.py` 通过校准数据集统计生成激活值的 scale，即 `max(abs(activation))`，方法也是类似的通过添加 hook 函数在遍历校准集的过程中计算激活值中的 max 值并记录到 `act_scales` 中。

![图片](/images/wechat/gptq-and-smoothquant-and-awq/006-55256259.jpg)

### 2. smooth 模型

接着再使用 export_int8_model.py 使用激活值 scale 和浮点精度模型生成量化精度模型：

1. 加载 FP16 模型
2. 加载激活值 scale
3. 使用激活值 scale smooth FP16 模型

![图片](/images/wechat/gptq-and-smoothquant-and-awq/007-a949b542.jpg)

最能体现论文思想的应该是其中第 3 步 smooth 部分，这是一个 attention 前的 laynorm + attention 的 smooth 实现，计算出 smooth scale 后对激活值的缩放前置到前面的 layernorm 层的 weights/bias 中，再对 fc 的 weight 乘以 scales，由此完成激活值和权重的平滑，对应论文中这个公式。第 4 步重新计算激活值 scale 和第 3 步类似。

![图片](/images/wechat/gptq-and-smoothquant-and-awq/008-b1953144.jpg)

![图片](/images/wechat/gptq-and-smoothquant-and-awq/009-87ea1e4d.jpg)

![图片](/images/wechat/gptq-and-smoothquant-and-awq/010-3bc40cc3.jpg)

### 3. 量化模型

最后使用 smooth 后的模型进行量化：

4. 使用 smooth 后的模型重新计算激活值 scale。
5. 使用 smooth 后模型和重新计算的激活值 scale 生成 INT8 模型

![图片](/images/wechat/gptq-and-smoothquant-and-awq/011-b4881b75.jpg)

export_int8_model.py

第 5 步中生成的 Int8OPTForCausalLM 是基于一些自定义 layer 实现的，如下所示，完整代码见 <https://github.com/mit-han-lab/smoothquant/blob/main/smoothquant/opt.py>。这些 layer 在项目 <https://github.com/Guangxuan-Xiao/torch-int> 中定义和实现，底层使用 CUTLASS 的 API 实现 Linear 和 BMM，属于比较典型的用法，CUTLASS 使用可以参考这篇文章进击的 Killua：CUTLASS 基础介绍（<https://zhuanlan.zhihu.com/p/671324125>）。

![图片](/images/wechat/gptq-and-smoothquant-and-awq/012-e00a5033.jpg)

opt.py

![图片](/images/wechat/gptq-and-smoothquant-and-awq/013-781aa1c6.jpg)

linear.cu

## 三、AWQ

AWQ（<https://arxiv.org/abs/2306.00978>）是一种 LLM 低比特权重量化方法，可以认为是当前 SOTA，已经被应用到很多低比特量化框架中。AWQ 关注在 low bit(INT3/INT4) weight 量化（W4A16），主要被应用在 linear layer（包含最多的参数）。它核心的贡献：

1. 发现 weight 对模型的重要程度存在极强的不均衡性，1% 的参数可能主导量化过程中损失的性能，假如我们在量化中保护这 1% 的参数，就能极大程度保护模型性能不受影响，但是混合精度（FP16+ 低比特）对硬件不友好。
2. 用激活值来发现重要 weight。
3. 对 weight 进行 per-channel 的 scale 同时对激活值除以 scale 来保护 weight。
4. 取和激活值相关的值进行 grid search，找到那个让量化误差最小的 scale。

![图片](/images/wechat/gptq-and-smoothquant-and-awq/014-422b9ae0.jpg)

本文围绕官方代码库（<https://github.com/mit-han-lab/llm-awq/tree/main>）进行算法实现的讲解，我们拆成 3 个部分来讲解，分别是：

1. 激活感知的 weight 缩放、扩大调整。
2. 权重量化。
3. 量化层推理。

### 1. 激活感知的 weight 缩放、扩大调整

根据前文描述在 weight 量化前，我们需要使用激活值对模型的原始 weight 进行调整，然后再进行第二步实际的量化，weight 缩放调整的完整代码见链接。为了简洁性本文基于 Llama 3 8B 模型来进行代码讲解，先来回顾下 Llama 3 的模型结构，首先是 embedding 层，紧接着是 32 层 DecoderLayer，最后是 Linear 层的 llm_head 输出，比较清晰。

```text
model LlamaForCausalLM(
  (model): LlamaModel(
    (embed_tokens): Embedding(128256, 4096)
    (layers): ModuleList(
      (0-31): 32 x LlamaDecoderLayer(
        (self_attn): LlamaSdpaAttention(
          (q_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (k_proj): Linear(in_features=4096, out_features=1024, bias=False)
          (v_proj): Linear(in_features=4096, out_features=1024, bias=False)
          (o_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (rotary_emb): LlamaRotaryEmbedding()
        )
        (mlp): LlamaMLP(
          (gate_proj): Linear(in_features=4096, out_features=14336, bias=False)
          (up_proj): Linear(in_features=4096, out_features=14336, bias=False)
          (down_proj): Linear(in_features=14336, out_features=4096, bias=False)
          (act_fn): SiLU()
        )
        (input_layernorm): LlamaRMSNorm()
        (post_attention_layernorm): LlamaRMSNorm()
      )
    )
    (norm): LlamaRMSNorm()
  )
  (lm_head): Linear(in_features=4096, out_features=128256, bias=False)
)
```

要利用激活值首先得准备一份校准数据生成并记录激活值内容，下面这段代码就是获取 LlamaDecoderLayer 第一层的输入数据和参数，作为后面逐层调整的输入数据。

```python
    layers = get_blocks(model)

    samples = get_calib_dataset(
        data=calib_data, tokenizer=enc, n_samples=n_samples, block_size=seqlen
    )
    samples = torch.cat(samples, dim=0)

    inps = []
    layer_kwargs = {}

    layers[0] = layers[0].cuda()
    move_embed(model, "cuda")

    # get input and kwargs to layer 0
    # with_kwargs is only supported in PyTorch 2.0
    # use this Catcher hack for now
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps.append(inp)
            layer_kwargs.update(kwargs)
            raise ValueError  # early exit to break later inference

    # patch layer 0 to catch input and kwargs
    layers[0] = Catcher(layers[0])
    try:
        model(samples.to(next(model.parameters()).device))
    except ValueError:  # work with early exit
        pass
    del samples
    layers[0] = layers[0].module  # restore
    inps = inps[0]

    layers[0] = layers[0].cpu()
    move_embed(model, "cpu")

    gc.collect()
    torch.cuda.empty_cache()
```

接下来就是逐层地去计算需要调整的 weight，每一层的输出会作为下一层的输入，在 LlamaDecoderLayer 内部使用 hook 的方式来记录每一线性子层的 input_feature，和 GPTQ 的做法类似。

```python
        layer = layers[i]
        layer = layer.cuda()
        named_linears = get_named_linears(layer)

        # firstly, get input features of all linear layers
        def cache_input_hook(m, x, y, name, feat_dict):
            x = x[0]
            x = x.detach().cpu()
            feat_dict[name].append(x)

        input_feat = defaultdict(list)
        handles = []
        for name in named_linears:
            handles.append(
                named_linears[name].register_forward_hook(
                    functools.partial(cache_input_hook, name=name, feat_dict=input_feat)
                )
            )
        inps = inps.to(next(layer.parameters()).device)  # in case multi-gpu
        # get output as next layer's input
        inps = layer(inps, **layer_kwargs)[0]
        for h in handles:
            h.remove()

        # now solve for scaling and clipping
        input_feat = {k: torch.cat(v, dim=0) for k, v in input_feat.items()}

        # Clear GPU memory
        torch.cuda.empty_cache()
```

然后就可以使用 input_feature 针对每个线性层进行 scale 计算，对于 llama3 模型根据权重和激活值的关系拆成 4 个子步骤来进行依次处理，分别是 [q_proj,k_proj,v_proj]，[o_proj]，[gate_proj,up_proj]，[down_proj]。

```python
    elif isinstance(module, LlamaDecoderLayer):
        # attention input
        scales_list.append(
            _auto_get_scale(
                prev_op=module.input_layernorm,
                layers=[
                    module.self_attn.q_proj,
                    module.self_attn.k_proj,
                    module.self_attn.v_proj,
                ],
                inp=input_feat["self_attn.q_proj"],
                module2inspect=module.self_attn,
                kwargs=module_kwargs,
            )
        )
        # attn out
        # Please refer to https://github.com/mit-han-lab/llm-awq/pull/67#issue-1850622696
        if module.self_attn.v_proj.weight.shape == module.self_attn.o_proj.weight.shape:
            scales_list.append(
                _auto_get_scale(
                    prev_op=module.self_attn.v_proj,
                    layers=[module.self_attn.o_proj],
                    inp=input_feat["self_attn.o_proj"],
                )
            )
        # fc1
        scales_list.append(
            _auto_get_scale(
                prev_op=module.post_attention_layernorm,
                layers=[module.mlp.gate_proj, module.mlp.up_proj],
                inp=input_feat["mlp.gate_proj"],
                module2inspect=module.mlp,
            )
        )
        # fc2
        scales_list.append(
            _auto_get_scale(
                prev_op=module.mlp.up_proj,
                layers=[module.mlp.down_proj],
                inp=input_feat["mlp.down_proj"],
            )
        )
```

在 \_auto_get_scale 中主要是调用 \_search_module_scale 进行 grid_search 找到最合适的 scale，使得调整权重 + 伪量化后损失最少，对应于论文这个公式，核心的代码如下所示，这部分的代码实现还是比较简洁的，其中 `w_quantize_func` 量化的部分在下个 part 介绍。

![图片](/images/wechat/gptq-and-smoothquant-and-awq/015-1b66e355.jpg)

scale 求解空间

```python
    def _search_module_scale(block, linears2scale: list, x, kwargs={}):
        # w: co, ci
        # x: n, ci
        x = x.to(next(block.parameters()).device)
        with torch.no_grad():
            org_out = block(x, **kwargs)
            if isinstance(org_out, tuple):
                org_out = org_out[0]
        x_max = get_act_scale(x)
        best_error = float("inf")
        best_ratio = -1
        best_scales = None

        n_grid = 20
        history = []

        org_sd = {k: v.cpu() for k, v in block.state_dict().items()}
        for ratio in range(n_grid):
            ratio = ratio * 1 / n_grid
            scales = x_max.pow(ratio).clamp(min=1e-4).view(-1)
            scales = scales / (scales.max() * scales.min()).sqrt()

            for fc in linears2scale:
                fc.weight.mul_(scales.view(1, -1).to(fc.weight.device))
                fc.weight.data = w_quantize_func(fc.weight.data) / (scales.view(1, -1))
            out = block(x, **kwargs)
            if isinstance(out, tuple):
                out = out[0]

            loss = (
                (org_out - out).float().pow(2).mean().item()
            )  # float prevents overflow
            history.append(loss)
            is_best = loss < best_error
            if is_best:
                best_error = loss
                best_ratio = ratio
                best_scales = scales
            block.load_state_dict(org_sd)
        if best_ratio == -1:
            print(history)
            raise Exception
        # print(best_ratio)
        best_scales = best_scales.view(-1)

        assert torch.isnan(best_scales).sum() == 0, best_scales
        return best_scales.detach()
```

scale 计算完成后，需要把它应用在每个线性层和它的前一层上，针对 layernorm+linear 和 linear+linear 的不同组合处理上大体类似，这里给出 ln+linear 的例子，可以看到 ln 层的 weight 和 bias 都除以了 scale，linear 层的 weight 都乘以了 scale，由此便完成了模型权重的调整。

```python
def scale_ln_fcs(ln, fcs, scales):
    if not isinstance(fcs, list):
        fcs = [fcs]

    scales = scales.to(ln.weight.device)

    ln.weight.div_(scales)
    if hasattr(ln, "bias") and ln.bias is not None:
        ln.bias.div_(scales)

    for fc in fcs:
        fc.weight.mul_(scales.view(1, -1))

    for p in ln.parameters():
        assert torch.isnan(p).sum() == 0
    for fc in fcs:
        for p in fc.parameters():
            assert torch.isnan(p).sum() == 0
```

### 2. 权重量化

在权重完成调整后就可以开始进行量化了，AWQ 也是逐层对 Linear 层进行权重量化，主体流程如下：

1. 先伪量化得到伪量化的权重、量化 scales 和 zeropoint，这里最重要的是用于后续 per-channel scales 和 zeropoint
2. 利用 scales 和 zero 来创建自定义的量化线性层 Module `WQLinear`，把模型中的 `Linear` 层替换为 `WQLinear` 层。

```text
                module.cuda()
                module.weight.data, scales, zeros = pseudo_quantize_tensor(
                    module.weight.data, n_bit=w_bit, get_scale_zp=True, **q_config
                )
                q_linear = WQLinear.from_linear(
                    module, w_bit, q_config["q_group_size"], False, scales, zeros
                )
                module.cpu()

                q_linear.to(next(layer.parameters()).device)
                set_op_by_name(layer, name, q_linear)
                torch.cuda.empty_cache()
                gc.collect()
```

伪量化的实现中规中矩，这里用的还是 min-max 方法来计算 scales，值得注意的是这里可以指定量化的 group_size 从而把计算 min-max 的范围控制的更小，这样有利用保持精度但同时对计算量要求更大了。

```python
def pseudo_quantize_tensor(
    w, n_bit=8, zero_point=True, q_group_size=-1, inplace=False, get_scale_zp=False
):
    org_w_shape = w.shape
    if q_group_size > 0:
        assert org_w_shape[-1] % q_group_size == 0
        w = w.reshape(-1, q_group_size)

    assert w.dim() == 2
    if zero_point:
        max_val = w.amax(dim=1, keepdim=True)
        min_val = w.amin(dim=1, keepdim=True)
        max_int = 2**n_bit - 1
        min_int = 0
        scales = (max_val - min_val).clamp(min=1e-5) / max_int
        zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)
    else:  # we actually never used this
        assert min_val is None
        max_val = w.abs().amax(dim=1, keepdim=True)
        max_val = max_val.clamp(min=1e-5)
        max_int = 2 ** (n_bit - 1) - 1
        min_int = -(2 ** (n_bit - 1))
        scales = max_val / max_int
        zeros = 0

    assert torch.isnan(scales).sum() == 0
    assert torch.isnan(w).sum() == 0

    if inplace:
        (
            (w.div_(scales).round_().add_(zeros)).clamp_(min_int, max_int).sub_(zeros)
        ).mul_(scales)
    else:
        w = (
            torch.clamp(torch.round(w / scales) + zeros, min_int, max_int) - zeros
        ) * scales
    assert torch.isnan(w).sum() == 0

    w = w.reshape(org_w_shape)

    if get_scale_zp:
        return w, scales.view(w.shape[0], -1), zeros.view(w.shape[0], -1)
    else:
        return w
```

得到 scale 和 zero 后就可以对浮点权重进行真正的量化并保存 4bit 的量化结果，这里复杂的不是量化过程而是量化后 4bit pack 保存的环节，即代码中量化后的 int32 类型的 `intweight` 到 int16 类型的 `awq_linear.qweight` 转换，是通过 `pack_intweight` 函数完成的。

```text
        intweight = []
        for idx in range(awq_linear.in_features):
            intweight.append(
                torch.round(
                    (linear.weight.data[:, idx] + scale_zeros[:, idx // group_size])
                    / qscales[:, idx // group_size]
                ).to(torch.int)[:, None]
            )
        intweight = torch.cat(intweight, dim=1)
        # intweight = intweight.t().contiguous()
        intweight = intweight.to(dtype=torch.int32)
        awq_linear.qweight = pack_intweight(
            intweight.contiguous(), interleave=4, kstride=64
        )
```

实现 `pack_intweight` 函数的开发应该是个 pytorch 好手，通过一系列的 `reshape`、`transpose` 和或运算把 int32 结果作为 int4 编码压缩到了 int16 的存储格式中，代码如下所示。这里给出了一个简单数据示例，通过这种方式存储的 `qweight` 在后续加载过程中可以一次高效地由 float4(128bit) 格式指令读取 32 个 int4 权重进行反量化和矩阵乘计算。

```python
def pack_intweight(unpacked_qweight, interleave, kstride):
    # unpacked_qweight: [N, K]
    N = unpacked_qweight.shape[0]
    K = unpacked_qweight.shape[1]

    Packed_Kernel = unpacked_qweight.cpu().numpy().reshape(N, K // 32, 32)
    # np.arange(32).reshape(4, 4, 2).transpose(1, 0, 2) => [0, 1, 8, 9, 16, 17, 24, 25, ...]
    Packed_Kernel = Packed_Kernel.reshape(N, K // 32, 4, 4, 2).transpose(0, 1, 3, 2, 4)
    Packed_Kernel = Packed_Kernel.reshape(N, K // 32, 32)

    # reorder each 8 weights for fast dequantization
    # [0, 1, 2, 3, 4, 5, 6, 7] => [0, 2, 4, 6, 1, 3, 5, 7]
    Packed_Kernel = Packed_Kernel.reshape(N, K // 32, 4, 8)
    Packed_Kernel = Packed_Kernel.reshape(N, K // 32, 4, 4, 2).transpose(0, 1, 2, 4, 3)
    Packed_Kernel = Packed_Kernel.reshape(N, K)

    # interleaving every four rows
    Packed_Kernel = Packed_Kernel.reshape(
        N // interleave, interleave, K // kstride, kstride
    )
    # N // 4, K // 64, 4, 64
    Packed_Kernel = Packed_Kernel.transpose(0, 2, 1, 3)
    Packed_Kernel = Packed_Kernel.reshape(
        N // interleave, K // kstride, kstride, interleave
    )
    # Packing -> (N // 4, K // 64, 64)

# >>> pack[...,0]
# array([[[  0,   1,   2,   3,   4,   5,   6,   7,  32,  33,  34,  35,
#           36,  37,  38,  39,  64,  65,  66,  67,  68,  69,  70,  71,
#           96,  97,  98,  99, 100, 101, 102, 103, 128, 129, 130, 131,
#          132, 133, 134, 135, 160, 161, 162, 163, 164, 165, 166, 167,
#          192, 193, 194, 195, 196, 197, 198, 199, 224, 225, 226, 227,
#          228, 229, 230, 231]]])
# >>> pack[...,1]
# array([[[  8,   9,  10,  11,  12,  13,  14,  15,  40,  41,  42,  43,
#           44,  45,  46,  47,  72,  73,  74,  75,  76,  77,  78,  79,
#          104, 105, 106, 107, 108, 109, 110, 111, 136, 137, 138, 139,
#          140, 141, 142, 143, 168, 169, 170, 171, 172, 173, 174, 175,
#          200, 201, 202, 203, 204, 205, 206, 207, 232, 233, 234, 235,
#          236, 237, 238, 239]]])
# >>> pack[...,2]
# array([[[ 16,  17,  18,  19,  20,  21,  22,  23,  48,  49,  50,  51,
#           52,  53,  54,  55,  80,  81,  82,  83,  84,  85,  86,  87,
#          112, 113, 114, 115, 116, 117, 118, 119, 144, 145, 146, 147,
#          148, 149, 150, 151, 176, 177, 178, 179, 180, 181, 182, 183,
#          208, 209, 210, 211, 212, 213, 214, 215, 240, 241, 242, 243,
#          244, 245, 246, 247]]])
# >>> pack[...,3]
# array([[[ 24,  25,  26,  27,  28,  29,  30,  31,  56,  57,  58,  59,
#           60,  61,  62,  63,  88,  89,  90,  91,  92,  93,  94,  95,
#          120, 121, 122, 123, 124, 125, 126, 127, 152, 153, 154, 155,
#          156, 157, 158, 159, 184, 185, 186, 187, 188, 189, 190, 191,
#          216, 217, 218, 219, 220, 221, 222, 223, 248, 249, 250, 251,
#          252, 253, 254, 255]]])

    Packed_Kernel = (
        Packed_Kernel[..., 0]
        | (Packed_Kernel[..., 1] << 4)
        | (Packed_Kernel[..., 2] << 8)
        | (Packed_Kernel[..., 3] << 12)
    )
    # reshape to (N // 4, K), FP16 format
    Packed_Kernel = Packed_Kernel.reshape(N // interleave, K)
    qweight = (
        torch.tensor(Packed_Kernel.astype("int16"))
        .to(unpacked_qweight.device)
        .contiguous()
    )
    return qweight
```

### 3. 量化层推理

在加载了量化后的 WQLinear 表示后就可以进行实际推理了，代码库中实现了相应的 CUDA Kernel 算子来加速推理过程，这里以 `gemv_forward_cuda_new` 举例来说明，这个函数实现了量化后 Int4 权重和向量乘积的结果，代码中的注释非常详细可读性很好，它的实现参考了 TensorRT-LLM（<https://github.com/NVIDIA/TensorRT-LLM/tree/d37b507f41a87457fe9f10f7459d08f5db235745/cpp/tensorrt_llm/kernels/weightOnlyBatchedGemv>）中的代码，算是比较中规中矩。其中反量化函数 `dequantize_s4_to_fp16x2` 的实现也没有重复造轮子，参考了 FasterTransformer（<https://github.com/NVIDIA/FasterTransformer/blob/main/src/fastertransformer/cutlass_extensions/include/cutlass_extensions/interleaved_numeric_conversion.h>）中 cutlass_extention 关于重叠格式转换（s4_to_fp16x2）的代码，几乎全是内联汇编指令，以后有需要也可以借鉴借鉴，完整代码详见 <https://github.com/mit-han-lab/llm-awq/blob/main/awq/kernels/csrc/quantization/dequantize.cuh>。

```cpp
template <int NPerBlock, int Batch, int BlockSize, int GroupSize>
__global__ void gemv_kernel(
  const half* inputs, const uint32_t* weight, const half* scales, const half* zeros, half* outputs,
  const int IC, const int OC)
{
    const int kStride = 64;
    const int kElemsPerThread = MEM_ACCESS_SIZE / 4;
    const int kThreadsNumPerTile = kStride / kElemsPerThread;
    // assert(MEM_ACCESS_SIZE == 128);

    static constexpr int kShuffleSize = 32;
    static constexpr int kShuffleBasicTile = 2;
    static constexpr int kShuffleContinous = 4;
    static constexpr int kShuffleStrided = 4;

    constexpr int Num = NPerBlock * Batch;
    constexpr int kInterleave = 4;

    half local_inputs[kElemsPerThread];
    uint32_t local_qweights[MEM_ACCESS_SIZE / 32];
    half half_weight_buffer[kElemsPerThread];
    half dequantized_weight[kElemsPerThread * NPerBlock];
    half local_scale[NPerBlock];
    half local_scaled_zeros[NPerBlock];

    half psum[Num];
    for (int i = 0; i < Num; ++i)
        psum[i] = static_cast<half>(0.f);

    extern __shared__ uint8_t shmem[];
    float(*out_smem)[Num * kInterleave] = reinterpret_cast<float(*)[Num * kInterleave]>(shmem);

    const int blk_row_offset = blockIdx.x * NPerBlock * kInterleave;
    const int thd_row_offset = (threadIdx.x / kThreadsNumPerTile) % kInterleave;
    const int act_k_offset = threadIdx.x / (kThreadsNumPerTile * kInterleave) * kStride
                               + (threadIdx.x % kThreadsNumPerTile) * kElemsPerThread;
    const int group_offset = act_k_offset / GroupSize;
    // TODO: use make_divisible
    const uint32_t* blk_weight_ptr = weight + blk_row_offset * IC / PACK_FACTOR;
    const half* scale_ptr = scales + blk_row_offset + thd_row_offset + group_offset * OC;
    const half* zeros_ptr = zeros + blk_row_offset + thd_row_offset + group_offset * OC;
    const half* inputs_ptr = inputs + act_k_offset;

    const int act_forward_step = BlockSize * kElemsPerThread / kInterleave;
    const int scale_forward_step = act_forward_step / GroupSize * OC;

    // Main loop iteration, each block completes the outputs for several OCs
    for (int kk = threadIdx.x * kElemsPerThread; kk < IC * kInterleave; kk += BlockSize * kElemsPerThread)
    {
        // Load qweight, scales and scaled_zeros
        #pragma unroll
        for (int idx = 0; idx < NPerBlock; ++idx)
        {
            // use float4 to load weights, each thread load 32 int4 numbers (1 x float4, 128 bit)
            *((float4*)(local_qweights)) =
                *((float4*)(blk_weight_ptr + (idx * kInterleave * IC + kk)/ PACK_FACTOR));
            local_scale[idx] = *(scale_ptr + idx * kInterleave);
            local_scaled_zeros[idx] = *(zeros_ptr + idx * kInterleave);

            // Map int4 qweight to fp format
            #pragma unroll
            for (int i = 0; i < MEM_ACCESS_SIZE / 32; ++i)
            {
                // Converts 32 bits (8 x int4) to 8 fp16
                dequantize_s4_to_fp16x2(*reinterpret_cast<half2 *>(local_qweights + i), reinterpret_cast<uint4 *>(half_weight_buffer + i * PACK_FACTOR));
            }

            // Dequantize (apply s/z) and shuffle elements to match the weight packing format
            #pragma unroll
            for (int i = 0; i < kShuffleContinous; ++i)
            {
                #pragma unroll
                for (int j = 0; j < kShuffleStrided; ++j)
                {
                    half2 w =
                        *reinterpret_cast<half2*>(
                          half_weight_buffer + (i + j * kShuffleContinous)* kShuffleBasicTile
                        );
                    w = __hfma2(w, __half2half2(local_scale[idx]), __half2half2(local_scaled_zeros[idx]));
                    dequantized_weight[((i * kShuffleStrided + j) * kShuffleBasicTile + 0)
                          * NPerBlock + idx]
                        = w.x;
                    dequantized_weight[((i * kShuffleStrided + j) * kShuffleBasicTile + 1)
                            * NPerBlock + idx]
                        = w.y;
                }
            }
        }
        #pragma unroll
        for (int batch_idx = 0; batch_idx < Batch; ++batch_idx)
        {
            const half* local_inputs_ptr = inputs_ptr + batch_idx * IC;
            #pragma unroll
            for (int idx = 0; idx < kElemsPerThread / 8; ++idx)
            {
                // load activation, 8 halves (128 bits) / step.
                *((float4*)(local_inputs + idx * 8)) = *((float4*)(local_inputs_ptr + idx * 8));
            }
            // Perform the MACs
            #pragma unroll
            for (int x = 0; x < NPerBlock / 2; ++x)
            {
                #pragma unroll
                for (int y = 0; y < kElemsPerThread; ++y)
                {
                    *reinterpret_cast<half2*>(psum + batch_idx * NPerBlock + x * 2)
                        = __hfma2(*reinterpret_cast<half2*>(dequantized_weight + y * NPerBlock + x * 2),
                            __half2half2(local_inputs[y]),
                            *reinterpret_cast<half2*>(psum + batch_idx * NPerBlock + x * 2));
                }
            }
        }
        inputs_ptr += act_forward_step;
        scale_ptr += scale_forward_step;
        zeros_ptr += scale_forward_step;
    }

    warp_reduce<Num, WARP_SIZE>(psum, out_smem);

    // Num * Interleave = batch * NPerBlock * Interleave -> 1 thread_block write back num
    for (int i = threadIdx.x; i < Num * kInterleave; i += BlockSize)
    {
        int batch_idx = i / (NPerBlock * kInterleave);
        int oc_idx = i % (NPerBlock * kInterleave);
        float acc = 0.f;
        for (int j = 0; j < BlockSize / WARP_SIZE; ++j)
        {
            acc += out_smem[j][i];
        }
        outputs[batch_idx * OC + blk_row_offset + oc_idx] = static_cast<half>(acc);
    }
}
```

笔者在 Llama3 8B 模型上测 wikitext 数据集，量化后 PPL 从 6.135(FP16) 上升到 6.532(INT4+g128)，比论文中 Llama2 的效果要差一些；GTX-4090+CUDA12.2 上单卡推理耗时 0.3224 秒下降到 0.2276 秒，效果看着还行。

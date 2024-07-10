# LayerNorm

간단한 튜토리얼. 모델의 한 예로 LayerNorm이 어떻게 처리되는지 살펴보겠습니다. 먼저 [PyTorch LayerNorm 문서](https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html)를 확인합니다. LayerNorm은 [Ba et al. 2016](https://arxiv.org/abs/1607.06450)의 원본 논문에서 유래되었으며, [Vaswani et al.](https://arxiv.org/abs/1706.03762)의 유명한 논문 Attention is All You Need에서 Transformer에 통합되었습니다. [GPT-2](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)는 Transformer와 동일한 아키텍처를 채택했지만, LayerNorm의 위치가 사전 정규화(pre-normalization) 버전으로 이동했습니다. 즉, Transformer의 잔여 경로(residual path)는 깨끗하게 유지되고, LayerNorm은 이제 Transformer의 각 블록의 첫 번째 레이어가 되었습니다. 이는 훈련 안정성을 긍정적으로 향상시킵니다.

[PyTorch LayerNorm](https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html)을 볼 때 첫 번째로 주목할 점은 실제로 방정식의 구현을 찾기 어려울 것이라는 점입니다. 이는 코드의 30단계 깊이에 숨겨져 있으며, 난해한 동적 디스패처 뒤에 있으며, 일부 자동 생성된 CUDA 코드에 있을 수 있습니다(자세한 내용은 [layer_norm.cpp](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/layer_norm.cpp) 및 [layer_norm_kernel.cu](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/layer_norm_kernel.cu)를 참조하십시오). 이는 PyTorch가 효율성을 매우 중요하게 생각하기 때문에 이루어집니다. 하지만 우리의 목적을 위해서는 먼저 더 간단한 PyTorch 연산을 사용하여 LayerNorm을 수동으로 구현해야 합니다. 이는 `LayerNorm` 모듈을 전달하는 것보다 훨씬 비효율적이지만 알고리즘적으로 유익합니다. 따라서 더 간단한 PyTorch 연산을 사용하여 LayerNorm의 수학적 구현은 다음과 같습니다:

```python
import torch
eps = 1e-5

class LayerNorm:

    @staticmethod
    def forward(x, w, b):
        # x는 입력 활성화, 형태는 B,T,C
        # w는 가중치, 형태는 C
        # b는 바이어스, 형태는 C
        B, T, C = x.size()
        # 평균 계산
        mean = x.sum(-1, keepdim=True) / C # B,T,1
        # 분산 계산
        xshift = x - mean # B,T,C
        var = (xshift**2).sum(-1, keepdim=True) / C # B,T,1
        # 역표준편차 계산: **0.5는 sqrt, **-0.5는 1/sqrt
        rstd = (var + eps) ** -0.5 # B,T,1
        # 입력 활성화 정규화
        norm = xshift * rstd # B,T,C
        # 정규화된 활성화를 스케일링하고 이동
        out = norm * w + b # B,T,C

        # 출력과 나중에 역전파에서 필요한 변수 캐시 반환
        cache = (x, w, mean, rstd)
        return out, cache
```

Transformer의 잔여 경로의 활성화 텐서는 훈련 중에 `B,T,C` 형태의 3차원 배열(텐서)입니다. 여기서 B는 배치 크기, T는 시간, C는 채널입니다. 예를 들어, B=8, T=1024, C=768은 가장 작은 GPT-2 모델(1억 2400만 매개변수)에서 볼 수 있는 설정 중 하나입니다.

이 레이어를 임의의 숫자로 전달할 수 있습니다:

```python
B = 2 # 일부 장난감 숫자
T = 3
C = 4
x = torch.randn(B, T, C, requires_grad=True)
w = torch.randn(C, requires_grad=True)
b = torch.randn(C, requires_grad=True)
out, cache = LayerNorm.forward(x, w, b)
```

우리가 얻는 출력은 `B,T,C` 형태의 텐서 `out`이며, 각 C차원 "섬유" 활성화는 정규화되고, 마지막에 이 레이어의 가중치와 바이어스로 스케일링되고 이동됩니다. 중요한 점은 `cache`라는 변수를 반환한다는 것입니다. 이는 입력 활성화 `x`, 가중치 `w`, 평균 `mean`, 역표준편차 `rstd`의 튜플입니다. 이는 역전파 동안 필요한 모든 변수입니다.

PyTorch는 Autograd를 사용하여 이 레이어의 역전파를 수행할 수 있습니다. 먼저 이를 수행해 보겠습니다:

```python
dout = torch.randn(B, T, C)
fakeloss = (out * dout).sum()
fakeloss.backward()
```

여기서 우리는 `fakeloss`를 생성했으며, 이는 단순히 레이어노름의 모든 출력의 (임의의) 가중 조합을 취합니다. 이는 단순히 모든 `B,T,C` 숫자를 단일 스칼라 값(손실)로 투영하는 것입니다. 일반적으로 이는 모델의 손실이지만, 여기서는 단순히 가짜 손실을 수행하고 있습니다. 그런 다음 이 스칼라에 대해 `backward()`를 호출하면 PyTorch는 이 그래프의 모든 입력에 대한 모든 기울기를 계산합니다 - 즉, 입력 활성화 `x`, 가중치 `w`, 바이어스 `b`입니다. Autograd에 대해 잘 모른다면, 제 [micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0) 비디오를 시청하는 것이 좋습니다. 여기서 우리는 작은 autograd 엔진을 구축합니다. PyTorch autograd의 마법은 `.backward`를 호출한 후 `requires_grad=True`인 모든 텐서의 `.grad` 속성을 해당 텐서에 대한 손실의 기울기로 채운다는 것입니다. 이 기울기는 x, w, b의 모든 입력 숫자에 대한 손실의 기울기를 알려줍니다. 따라서 `x.grad`, `w.grad`, `b.grad`의 형태는 `x`, `w`, `b`의 형태와 정확히 동일합니다.

하지만 우리는 PyTorch Autograd를 사용하고 싶지 않습니다. 우리는 역전파를 수동으로 수행하고 싶습니다. 따라서 우리는 LayerNorm의 표현식을 작성합니다. 순전파는 다음과 같은 수학적 형태를 가집니다:

$\text{LayerNorm}(x) = w \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + b$

여기서 $\odot$는 요소별 곱셈, $\mu$는 평균, $\sigma^2$는 분산, $\epsilon$은 0으로 나누는 것을 방지하기 위한 작은 상수입니다. 미적분학의 미분 규칙을 기억하면서, 이제 기울기를 도출하고자 합니다. 이 부분에서는 [Becoming a Backprop Ninja](https://www.youtube.com/watch?v=q8SA3rM6ckI) 비디오가 매우 유용할 수 있습니다. 여기서 유사한 레이어인 배치 정규화 레이어를 자세히 다룹니다. 미분을 통해 표현식을 단순화하고 항을 이동하여 표현식을 약간 단순화할 수 있습니다. 따라서 순전파의 각 개별 라인을 수동으로 역전파할 필요는 없습니다. 특히, 우리는 다음과 같은 결과를 얻습니다:

```python
    @staticmethod
    def backward(dout, cache):
        x, w, mean, rstd = cache
        # norm을 다시 계산 (메모리를 절약하는 대신 계산 비용 증가)
        norm = (x - mean) * rstd
        # 가중치, 바이어스에 대한 기울기
        db = dout.sum((0, 1))
        dw = (dout * norm).sum((0, 1))
        # 입력에 대한 기울기
        dnorm = dout * w
        dx = dnorm - dnorm.mean(-1, keepdim=True) - norm * (dnorm * norm).mean(-1, keepdim=True)
        dx *= rstd
        return dx, dw, db
```

따라서 출력 숫자 `dout`에 대한 기울기와 순전파의 `cache`를 사용하여 이 레이어를 통해 입력으로 역전파할 수 있으며, 역전파의 연쇄 규칙을 계속할 수 있습니다. 이제 우리는 우리의 역전파를 수행하고 결과가 일치하는지 확인할 수 있습니다(오차는 매우 작습니다):

```python
dx, dw, db = LayerNorm.backward(dout, cache)
print("dx error:", (x.grad - dx).abs().max().item())
print("dw error:", (w.grad - dw).abs().max().item())
print("db error:", (b.grad - db).abs().max().item())
```

한 가지 더 주목할 점은 역전파에서 `norm` 변수를 다시 계산했다는 것입니다. 우리는 이미 순전파에서 이 변수를 계산했지만 버렸습니다! 이를 `cache`의 일부로 만들어 다시 계산하지 않을 수도 있었습니다. 실제로 그렇게 할 수 있으며, 동일한 결과를 얻을 수 있습니다. `cache`에 저장하는 양은 전적으로 우리에게 달려 있습니다. `mean`과 `rstd`도 저장하지 않고 역전파에서 다시 계산할 수 있었습니다. 차이점은 `mean`과 `rstd`는 매우 작으며, 형태는 `B,T`입니다. 반면 `norm`은 `B,T,C` 형태입니다. 따라서 이는 메모리와 계산 간의 단순한 트레이드오프입니다. `norm`을 캐시에 저장하지 않음으로써 메모리를 절약하지만, 나중에 역전파에서 약간의 계산을 교환합니다. 이는 모든 레이어에서 매우 일반적이며, 다양한 딥러닝 프레임워크의 레이어 구현은 모두 다른 "체크포인트 설정"을 가질 수 있습니다. 혼란스럽게도 이는 체크포인트라고 불리며, 모델 가중치를 디스크에 저장하는 것과는 아무런 관련이 없습니다. 이는 순전파에서 중간 변수를 저장하여 역전파에서 계산을 절약하는 것입니다.

이제 PyTorch 텐서를 사용한 버전을 살펴보았습니다. 이제 이를 C로 이동하고 Tensor 추상화를 제거해야 합니다. 순전파의 전체 구현을 제공하기 전에, 텐서에 대한 간단한 설명을 하겠습니다. 텐서란 무엇입니까? 텐서는 1) 원시 데이터를 저장하는 1차원 메모리 블록인 Storage와 2) 해당 저장소에 대한 View로 구성됩니다. [PyTorch Internals](http://blog.ezyang.com/2019/05/pytorch-internals/)가 도움이 될 수 있습니다. 예를 들어, 3차원 텐서가 있다고 가정해 보겠습니다:

```python
torch.manual_seed(42)
B, T, C = 2, 3, 4
a = torch.randn(B, T, C)
print(a)

tensor([[[ 1.9269,  1.4873,  0.9007, -2.1055],
         [ 0.6784, -1.2345, -0.0431, -1.6047],
         [ 0.3559, -0.6866, -0.4934,  0.2415]],

        [[-1.1109,  0.0915, -2.3169, -0.2168],
         [-0.3097, -0.3957,  0.8034, -0.6216],
         [-0.5920, -0.0631, -0.8286,  0.3309]]])
```

이는 2x3x4 텐서이지만, 기본 메모리는 단일 1차원 배열로 크기는 2\*3\*4=24입니다. View는 이 1차원 배열에 대한 형태일 뿐입니다. 이제 PyTorch 텐서에 인덱싱할 때, 예를 들어 `a[1,2,3]`, PyTorch는 1차원 배열의 오프셋을 `1*3*4 + 2*4 + 3 = 23`으로 계산하고 해당 오프셋의 값을 반환합니다. 일반적인 공식은 `b,t,c` 요소를 검색하려면 Storage의 오프셋을 `b*T*C + t*C + c`로 계산하는 것입니다. 예를 들어:

```python
b,t,c = 1,2,3
print(a[b,t,c])
print(a.view(-1)[b*T*C + t*C + c])
```

이 두 가지는 모두 0.3309를 출력합니다. 따라서 이 방식으로 모든 개별 요소에 접근하는 방법과 모든 포인터의 오프셋을 계산하는 방법을 알 수 있습니다. 특히 채널 차원이 가장 안쪽 차원임을 주목하십시오. 따라서 오프셋을 1씩 증가시키면 채널 차원을 순회하게 됩니다. 이는 C 구현의 메모리 레이아웃을 고려할 때 중요합니다. C에서의 순전파는 다음과 같습니다:

```c
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void layernorm_forward(float* out, float* mean, float* rstd,
                       float* inp, float* weight, float* bias,
                       int B, int T, int C) {
    float eps = 1e-5f;
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            // 입력 위치 inp[b,t,:]로 이동
            float* x = inp + b * T * C + t * C;
            // 평균 계산
            float m = 0.0f;
            for (int i = 0; i < C; i++) {
                m += x[i];
            }
            m = m/C;
            // 분산 계산 (바이어스 보정 없이)
            float v = 0.0f;
            for (int i = 0; i < C; i++) {
                float xshift = x[i] - m;
                v += xshift * xshift;
            }
            v = v/C;
            // rstd 계산
            float s = 1.0f / sqrtf(v + eps);
            // 출력 위치 out[b,t,:]로 이동
            float* out_bt = out + b * T * C + t * C;
            for (int i = 0; i < C; i++) {
                float n = (s * (x[i] - m)); // 정규화된 출력
                float o = n * weight[i] + bias[i]; // 스케일링 및 이동
                out_bt[i] = o; // 쓰기
            }
            // 역전파를 위해 평균 및 rstd 캐시
            mean[b * T + t] = m;
            rstd[b * T + t] = s;
        }
    }
}
```

포인터를 `inp[b,t]`로 오프셋하고, 다음 `C` 요소는 (배치, 시간) 위치의 채널임을 알 수 있습니다. 그리고 역전파:

```c
void layernorm_backward(float* dinp, float* dweight, float* dbias,
                        float* dout, float* inp, float* weight, float* mean, float* rstd,
                        int B, int T, int C) {
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            float* dout_bt = dout + b * T * C + t * C;
            float* inp_bt = inp + b * T * C + t * C;
            float* dinp_bt = dinp + b * T * C + t * C;
            float mean_bt = mean[b * T + t];
            float rstd_bt = rstd[b * T + t];

            // 첫 번째: 두 개의 reduce 연산
            float dnorm_mean = 0.0f;
            float dnorm_norm_mean = 0.0f;
            for (int i = 0; i < C; i++) {
                float norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
                float dnorm_i = weight[i] * dout_bt[i];
                dnorm_mean += dnorm_i;
                dnorm_norm_mean += dnorm_i * norm_bti;
            }
            dnorm_mean = dnorm_mean / C;
            dnorm_norm_mean = dnorm_norm_mean / C;

            // 이제 다시 반복하여 모든 기울기 누적
            for (int i = 0; i < C; i++) {
                float norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
                float dnorm_i = weight[i] * dout_bt[i];
                // 바이어스에 대한 기울기 기여
                dbias[i] += dout_bt[i];
                // 가중치에 대한 기울기 기여
                dweight[i] += norm_bti * dout_bt[i];
                // 입력에 대한 기울기 기여
                float dval = 0.0f;
                dval += dnorm_i; // 항 1
                dval -= dnorm_mean; // 항 2
                dval -= norm_bti * dnorm_norm_mean; // 항 3
                dval *= rstd_bt; // 최종 스케일링
                dinp_bt[i] += dval;
            }
        }
    }
}
```

One additional detail to note is that we always += into the gradients. We never use = and we never use *=. This is important stylistically because if you have one variable used multiple times in a graph, the backward pass gradients always add up. In this repo this is not important because we don't have exotic branching, but it's proper. So during training we always first do `zero_grad` to set all the gradients to zero, and then we accumulate into them during backward pass.

One more note on differences between training and inference. Some of you may have already seen my earlier project [llama2.c](https://github.com/karpathy/llama2.c), which inferences Llama 2 architecture in pure C. Unlike GPT-2, Llama 2 swaps out LayerNorm for the much simpler RMSNorm. You can see the implementation of the [RMSNorm in llama2.c](https://github.com/karpathy/llama2.c/blob/master/run.c#L182), copy pasting it here:

```c
void rmsnorm(float* o, float* x, float* weight, int size) {
    // calculate sum of squares
    float ss = 0.0f;
    for (int j = 0; j < size; j++) {
        ss += x[j] * x[j];
    }
    ss /= size;
    ss += 1e-5f;
    ss = 1.0f / sqrtf(ss);
    // normalize and scale
    for (int j = 0; j < size; j++) {
        o[j] = weight[j] * (ss * x[j]);
    }
}
```

How does this differ to our LayerNorm above?

- First, algorithmically, you'll notice that RMSNorm does not keep track of or subtract the mean, it only normalizes by the norm. Notice: norm, not standard deviation, because we did not subtract the mean. This is a simplification of the layer that has now become very trendy because it works just as well, if not slightly better. Also, the RMSNorm does not have biases, it only has a weight for scaling after normalization. In general, GPT-2 used way too many biases everywhere and it turns out you can remove these - from all the Linear Layers and from LayerNorms. The network can "simulate" biases if it needs them, e.g. by allocating one of the channel dimensions to be constant (data-independent), and then any weight multiplying that constant dimension will effectively work like a bias. This significantly simplies a lot of the code.
- Second, the inference code has no batch dimension B, i.e. the batch size is assumed to be 1. You could in principle have batched inference as well, especially if you wish to host an LLM that you expect many simultaneous queries to. But if you're just running an LLM locally, chances are you just want to have a single "stream" of generation, so there is no batch size for parallelism that could support multiple streams at once. To keep things simple, llama2.c is not batched, and therefore you won't see any loops that look like `for (int b = 0; b < B; b++)`.
- Third, this inference code has no time dimension T within this individual layer. During training, we can loop over time inside each layer and calculate the layernorm at all time steps. But during inference, we have to generate one token at a time, feeding the token predicted at time `t` into the forward pass of the Transformer at the next time step `t+1`. So this is why you don't see any loops that look like `for (int t = 0; t < T; t++)` inside individual layers. This loop over time [does exist](https://github.com/karpathy/llama2.c/blob/master/run.c#L747), but it is on the outside of the Transformer forward pass.
- You'll see that we don't keep track of any intermediate calculations, memory, or cache. That's because during inference, there is no `.backward` pass that will follow. We only need to calculate the output, and we don't need to keep any intermediate variables around. As a result, the memory consumption of inference is significantly lower than that of training. We can afford to just discard activations, and only keep memory for the "activation frontier". Similarly, there is no need to implement the `backward` function for this RMSNorm anywhere, as there is no backward pass.

As a result of all these difference, training is significantly more complex and involved, both algorithmically and computationally, and that's partly why I started by writing inference (llama2.c) before I implemented training (llm.c, here). Finally, I am attaching two helper files to this same directory that have the complete code. First:

```
python layernorm.py
```

To write out the reference data from PyTorch. Then compile and run the C version:

```
gcc layernorm.c -o layernorm -lm
./layernorm
```

You'll see that everything matches ok.

This was just the LayerNorm. We go through the exact same process for all the other layers. Most of the other layers are actually easier than LayerNorm. Hope that helps!

# llm.c

245MB의 PyTorch나 107MB의 cPython이 필요 없는 간단하고 순수한 C/CUDA로 작성된 LLMs. 현재 초점은 사전 훈련에 있으며, 특히 [GPT-2](https://github.com/openai/gpt-2) 및 [GPT-3](https://arxiv.org/abs/2005.14165) 미니 시리즈를 재현하는 것입니다. [train_gpt2.py](train_gpt2.py)에서 병렬 PyTorch 참조 구현과 함께 제공됩니다. 이 파일은 이전 프로젝트인 [nanoGPT](https://github.com/karpathy/nanoGPT)를 약간 수정한 것입니다. 현재 llm.c는 PyTorch Nightly보다 약 7% 더 빠릅니다. [train_gpt2.cu](train_gpt2.cu)에서 최신 메인라인 코드를 제공하며, 단일 파일 [train_gpt2.c](train_gpt2.c)에서 약 1,000줄의 깨끗한 코드로 작성된 간단한 참조 CPU fp32 구현도 있습니다. 이 저장소는 C와 CUDA 코드만 유지하고자 합니다. 다른 언어나 저장소로의 포트는 매우 환영하지만, 별도의 저장소에서 이루어져야 하며, "주목할 만한 포크" 섹션에서 링크를 제공할 수 있습니다. 개발자 조정은 [Discussions](https://github.com/karpathy/llm.c/discussions) 및 Discord, `#llmc` 채널에서 [Zero to Hero](https://discord.gg/3zy8kqD9Cp) 채널 또는 CUDA MODE Discord의 `#llmdotc`에서 이루어집니다.

## 빠른 시작

오늘날 llm.c 저장소에 대한 최고의 소개는 GPT-2 (124M) 모델을 재현하는 것입니다. [Discussion #481](https://github.com/karpathy/llm.c/discussions/481)에서 이를 자세히 설명합니다. 우리는 llm.c와 PyTorch의 병렬 구현에서 GPT-2 및 GPT-3 시리즈의 다른 모델을 재현할 수 있습니다. [scripts README](scripts/README.md)를 참조하십시오.

디버깅 팁: 바이너리를 빌드하기 위해 `make` 명령을 실행할 때 `-O3`를 `-g`로 교체하여 좋아하는 IDE(vscode 등)에서 코드를 단계별로 실행할 수 있습니다.

## 빠른 시작 (1 GPU, fp32 전용)

여러 노드에서 훈련하지 않고 혼합 정밀도에 관심이 없으며 CUDA를 배우고자 한다면, fp32 (레거시) 파일이 관심을 끌 수 있습니다. 이 파일들은 llm.c의 초기 역사에서 "체크포인트"된 파일로, 더 간단하고 이식성이 뛰어나며 이해하기 쉬울 수 있습니다. 1 GPU, fp32 코드를 실행하려면 다음과 같이 하십시오:

```bash
chmod u+x ./dev/download_starter_pack.sh
./dev/download_starter_pack.sh
make train_gpt2fp32cu
./train_gpt2fp32cu
```

download_starter_pack.sh 스크립트는 시작하기 쉽고 빠르게 .bin 파일을 다운로드하여 시작할 수 있도록 도와줍니다. 이 파일에는 1) fp32로 저장된 GPT-2 124M 모델, bfloat16, 2) 단위 테스트에 사용되는 "디버그 상태" (작은 배치 데이터, 대상 활성화 및 그래디언트), 3) GPT-2 토크나이저, 4) 토큰화된 [tinyshakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt) 데이터셋이 포함되어 있습니다. 또는 .sh 스크립트를 실행하는 대신 다음과 같이 수동으로 이러한 아티팩트를 재생성할 수 있습니다:

```bash
pip install -r requirements.txt
python dev/data/tinyshakespeare.py
python train_gpt2.py
```

## 빠른 시작 (CPU)

"GPU가 하나도 없는 사람들을 위한" 섹션입니다. 여전히 llm.c 훈련을 즐길 수 있습니다! 하지만 멀리 가지는 못할 것입니다. fp32 버전과 마찬가지로 CPU 버전은 llm.c의 초기 역사에서 단순한 참조 구현이었을 때의 체크포인트입니다. 예를 들어, 처음부터 훈련하는 대신 GPT-2 small (124M)을 초기화하여 Shakespeare와 같은 텍스트를 출력하도록 미세 조정할 수 있습니다. 예를 들어:

```bash
chmod u+x ./dev/download_starter_pack.sh
./dev/download_starter_pack.sh
make train_gpt2
OMP_NUM_THREADS=8 ./train_gpt2
```

시작 팩 스크립트를 실행하지 않으려면 이전 섹션에서 언급한 것처럼 `python dev/data/tinyshakespeare.py`를 실행한 다음 `python train_gpt2.py`를 실행하여 동일한 .bin 파일과 아티팩트를 재현할 수 있습니다.

위의 명령은 (1) 이미 토큰화된 [tinyshakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt) 데이터셋을 다운로드하고 GPT-2 (124M) 가중치를 다운로드합니다. (3) C에서 초기화하고 AdamW를 사용하여 tineshakespeare에서 40단계 동안 훈련하고 (배치 크기 4, 컨텍스트 길이 64만 사용), 검증 손실을 평가하고 일부 텍스트를 샘플링합니다. 솔직히 말해서, 강력한 CPU가 없으면 (그리고 OMP 스레드 수를 늘릴 수 있다면) CPU에서 LLM을 훈련하는 데 멀리 가지 못할 것입니다. 하지만 좋은 데모/참조가 될 수 있습니다. 내 MacBook Pro (Apple Silicon M3 Max)에서 출력은 다음과 같습니다:

```
[GPT-2]
max_seq_len: 1024
vocab_size: 50257
num_layers: 12
num_heads: 12
channels: 768
num_parameters: 124439808
train dataset num_batches: 1192
val dataset num_batches: 128
num_activations: 73323776
val loss 5.252026
step 0: train loss 5.356189 (took 1452.121000 ms)
step 1: train loss 4.301069 (took 1288.673000 ms)
step 2: train loss 4.623322 (took 1369.394000 ms)
step 3: train loss 4.600470 (took 1290.761000 ms)
... (trunctated) ...
step 39: train loss 3.970751 (took 1323.779000 ms)
val loss 4.107781
generating:
---
Come Running Away,
Greater conquer
With the Imperial blood
the heaviest host of the gods
into this wondrous world beyond.
I will not back thee, for how sweet after birth
Netflix against repounder,
will not
flourish against the earlocks of
Allay
---
```

## 데이터셋

`/dev/data/(dataset).py` 내부의 데이터 파일은 다운로드, 토큰화 및 토큰을 .bin 파일로 저장하는 역할을 합니다. 예를 들어 다음을 실행할 때:

```bash
python dev/data/tinyshakespeare.py
```

[tinyshakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt) 데이터셋을 다운로드하고 토큰화합니다. 출력은 다음과 같습니다:

```
writing 32,768 tokens to ./dev/data/tinyshakespeare/tiny_shakespeare_val.bin
writing 305,260 tokens to ./dev/data/tinyshakespeare/tiny_shakespeare_train.bin
```

.bin 파일에는 짧은 헤더(1024바이트)와 GPT-2 토크나이저로 토큰 ID를 나타내는 uint16 형식의 토큰 스트림이 포함되어 있습니다. 더 많은 데이터셋은 `/dev/data`에 있습니다.

## 테스트

C 코드가 PyTorch 코드와 일치하는지 확인하기 위한 간단한 단위 테스트도 첨부했습니다. 예를 들어 CPU에서 컴파일하고 실행하려면 다음과 같이 하십시오:

```bash
make test_gpt2
./test_gpt2
```

이제 `train_gpt2.py`에서 작성된 `gpt2_124M_debug_state.bin` 파일을 로드하고, 전방 패스를 실행하고, PyTorch 참조 구현과 로짓 및 손실을 비교한 다음, Adam으로 10회 반복 훈련을 수행하고 손실이 PyTorch와 일치하는지 확인합니다. GPU 버전을 테스트하려면 다음을 실행합니다:

```bash
# fp32 테스트 (cudnn 미지원)
make test_gpt2cu PRECISION=FP32 && ./test_gpt2cu
# 혼합 정밀도 cudnn 테스트
make test_gpt2cu USE_CUDNN=1 && ./test_gpt2cu
```

이 테스트는 fp32 경로와 혼합 정밀도 경로를 모두 테스트합니다. 테스트는 통과해야 하며 `overall okay: 1`을 출력해야 합니다.

## 튜토리얼

여기에 매우 작은 튜토리얼을 첨부했습니다. [doc/layernorm/layernorm.md](doc/layernorm/layernorm.md)에서 확인할 수 있습니다. 이는 GPT-2 모델의 단일 레이어인 layernorm 레이어를 구현하는 간단한 단계별 가이드입니다. C에서 레이어가 어떻게 구현되는지 이해하는 좋은 출발점입니다.

**플래시 어텐션**. 2024년 5월 1일부터 cuDNN의 플래시 어텐션을 사용합니다. cuDNN은 컴파일 시간을 몇 초에서 약 1분으로 늘리며, 이 코드 경로는 현재 매우 새롭기 때문에 기본적으로 비활성화되어 있습니다. 다음과 같이 컴파일하여 활성화할 수 있습니다:

```bash
make train_gpt2cu USE_CUDNN=1
```

이 명령은 cudnn으로 컴파일하고 실행하려고 시도합니다. 시스템에 cuDNN이 설치되어 있어야 합니다. [cuDNN 설치 지침](https://developer.nvidia.com/cudnn)을 참조하십시오. apt-get을 사용하면 기본 cuDNN 패키지 세트를 가져올 수 있습니다. 최소 설정을 위해서는 cuDNN dev 패키지만으로 충분합니다. 예를 들어 Ubuntu 22.04에서 CUDA 12.x를 사용하는 경우:

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install libcudnn9-dev-cuda-12
```

이 외에도 [cuDNN 프론트엔드](https://github.com/NVIDIA/cudnn-frontend/tree/main)가 필요하지만, 이는 헤더 파일에 불과합니다. 단순히 저장소를 디스크에 클론하십시오. Makefile은 현재 홈 디렉토리 또는 현재 디렉토리에서 이를 찾습니다. 다른 위치에 두었다면 `CUDNN_FRONTEND_PATH=/path/to/your/cudnn-frontend/include`를 `make` 명령줄에 추가하십시오.

## 다중 GPU 훈련

MPI 및 NCCL을 설치하십시오. 예를 들어 Linux에서:

```bash
sudo apt install openmpi-bin openmpi-doc libopenmpi-dev
```

NCCL은 [공식 웹사이트](https://developer.nvidia.com/nccl/nccl-download)의 지침을 따르십시오 (예: 네트워크 설치 프로그램)

그런 다음:

```bash
make train_gpt2cu
mpirun -np <number of GPUs> ./train_gpt2cu
```

또는 `./scripts/` 아래의 스크립트 중 하나를 실행하십시오.

## 다중 노드 훈련

[NCCL](#multi-gpu-training) 섹션의 지침을 따라 NCCL을 설치했는지 확인하십시오.

현재 지원하는 다중 노드 훈련을 실행하는 3가지 방법이 있습니다:
1) OpenMPI를 사용하여 nccl id를 교환하고 NCCL을 초기화합니다. 자세한 내용은 `./scripts/multi_node/run_gpt2_124M_mpi.sh` 스크립트를 참조하십시오.
2) 공유 파일 시스템을 사용하여 NCCL을 초기화합니다. 자세한 내용은 `./scripts/multi_node/run_gpt2_124M_fs.sbatch` 스크립트를 참조하십시오.
3) TCP 소켓을 사용하여 NCCL을 초기화합니다. 자세한 내용은 `./scripts/multi_node/run_gpt2_124M_tcp.sbatch` 스크립트를 참조하십시오.

참고:
* 슬럼 환경에서 실행 중이고 슬럼이 PMIx를 지원하지 않는 경우 (일반적인 상황일 것으로 예상됨) FS (2) 또는 TCP (3) 접근 방식을 사용해야 합니다. 슬럼이 PMIx를 지원하는지 테스트하려면: `srun --mpi=list`를 실행하고 출력에 `pmix`가 있는지 확인하십시오.
* 슬럼이 설정되어 있지 않은 경우 `mpirun`을 사용하여 다중 노드 실행을 시작할 수 있습니다 - MPI (1).

이 세 가지 방법 중 어느 것도 우월하지 않으며, 특정 환경에서 실행할 수 있도록 옵션을 제공합니다.

## 실험 / 스윕

TinyStories에서 4개의 GPU를 사용하여 학습률을 스윕하는 예제 프로세스입니다. 셸 스크립트 `sweep.sh`를 실행하십시오 (물론 `chmod u+x sweep.sh` 후):

```bash
#!/bin/bash

learning_rates=(3e-5 1e-4 3e-4 1e-3)

for i in {0..3}; do
    export CUDA_VISIBLE_DEVICES=$i
    screen -dmS "tr$i" bash -c "./train_gpt2cu -i data/TinyStories -v 250 -s 250 -g 144 -l ${learning_rates[$i]} -o stories$i.log"
done

# 다음 명령으로 종료할 수 있습니다
# screen -ls | grep -E "tr[0-3]" | cut -d. -f1 | xargs -I {} screen -X -S {} quit
```

이 예제는 4개의 화면 세션을 열고 다른 학습률로 네 가지 명령을 실행합니다. 이는 모든 손실을 포함한 로그 파일 `stories$i.log`를 작성하며, 이를 Python에서 원하는 대로 플롯할 수 있습니다. 이러한 로그 파일을 구문 분석하고 플롯하는 방법의 간단한 예는 [dev/vislog.ipynb](dev/vislog.ipynb)에 있습니다.

## 저장소

이 저장소가 되기를 원하는 몇 가지 사항:

첫째, `llm.c`가 교육의 장소가 되기를 원합니다. 예를 들어, `dev/cuda` 폴더는 모든 레이어에 대한 커널을 수동으로 작성하고 매우 잘 문서화된 라이브러리입니다. 매우 간단한 커널부터 더 복잡하고 빠른 커널까지 시작합니다. 다양한 트레이드오프가 있는 새로운 커널이 있다면, 여기에 기여해 주십시오.

그렇긴 하지만, `llm.c`가 매우 빠르기를 원하며, 심지어 네트워크를 훈련하는 데 실용적일 수 있기를 원합니다. 예를 들어, 시작으로 큰 GPT-2 (1.6B) 훈련 실행을 재현할 수 있어야 합니다. 이를 위해서는 cuBLAS, cuBLASLt, CUTLASS, cuDNN 등과 같은 라이브러리를 사용하는 것을 포함하여 가능한 가장 빠른 커널을 통합해야 합니다. 이렇게 하는 것이 전문가의 상한선을 설정하고 측정 단위를 설정하는 교육적 목적을 제공한다고 생각합니다. 예를 들어, 수동으로 작성한 커널이 cuBLAS 속도의 80%라고 말할 수 있습니다. 그런 다음 매우 빠른 실행을 선택하거나 원하는 수동 커널을 "드래그 앤 드롭"하여 실행할 수 있습니다.

그러나 제약 조건으로, 루트 폴더의 `llm.c`를 간단하고 읽기 쉽게 유지하고 싶습니다. 예를 들어, 성능을 2% 향상시키지만 500줄의 복잡한 C 코드와 이국적인 서드 파티 종속성이 필요한 PR은 복잡성이 가치가 없기 때문에 거부할 수 있습니다. 구체적인 예로, 루트 훈련 루프에서 matmul에 대해 cuBLAS를 기본값으로 설정하는 것은 당연한 일입니다. 이는 메인라인 코드를 훨씬 빠르게 만들고, 해석 가능한 단일 줄의 코드이며, 매우 일반적인 종속성입니다. 이와 별도로 `dev/cuda`에 수동 구현을 포함할 수 있습니다.

마지막으로, 프로젝트의 주요/기본 파일이 포함된 루트 폴더의 복잡성에 대해 더 민감할 것입니다. 비교적으로, `dev/` 폴더는 커널 또는 클래스를 개발하고 유용하거나 관련되거나 교육적인 코드를 공유하기 위한 스크래치 공간입니다. 이 코드 중 일부는 (로컬로) 복잡할 수 있습니다.

## 주목할 만한 포크

- AMD 지원
  - [llm.c](https://github.com/anthonix/llm.c) by @[anthonix](https://github.com/anthonix): 7900 XTX와 같은 AMD 장치를 지원합니다.

- C#
  - [llm.cs](https://github.com/azret/llm.cs) by @[azret](https://github.com/azret): 이 프로젝트의 C# 포트
  - [Llm.cs](https://github.com/nietras/Llm.cs) by @[nietras](https://github.com/nietras): 모든 플랫폼에서 쉽게 시작할 수 있도록 초점을 맞춘 이 프로젝트의 C# 포트. 클론하고 실행 ✅

- CUDA C++
  - [llm.cpp](https://github.com/gevtushenko/llm.c) by @[gevtushenko](https://github.com/gevtushenko): [CUDA C++ Core Libraries](https://github.com/NVIDIA/cccl)를 사용한 이 프로젝트의 포트
     - A presentation this fork was covered in [this lecture](https://www.youtube.com/watch?v=WiB_3Csfj_Q) in the [CUDA MODE Discord Server](https://discord.gg/cudamode)

- Go
  - [llm.go](https://github.com/joshcarp/llm.go) by @[joshcarp](https://github.com/joshcarp): a Go port of this project

- Java
  - [llm.java](https://github.com/harryjackson/llm.java) by @[harryjackson](https://github.com/harryjackson): a Java port of this project

- Metal
  - [llm.metal](https://github.com/regrettable-username/llm.metal) by @[regrettable-username](https://github.com/regrettable-username): LLM training in simple, raw C/Metal Shading Language

- Mojo
  - [llm.🔥](https://github.com/dorjeduck/llm.mojo) by @[dorjeduck](https://github.com/dorjeduck): a Mojo port of this project

- OpenCL
  - [llm.c](https://github.com/krrishnarraj/llm.c) by @[krrishnarraj](https://github.com/krrishnarraj): an OpenCL port of this project

- Rust
  -  [llm.rs](https://github.com/yijunyu/llm.rs) by @[Yijun Yu](https://github.com/yijunyu): a Rust rewrite with the aim to have same performance
  -  [llm.rs](https://github.com/ToJen/llm.rs) by @[ToJen](https://github.com/ToJen): a Rust port of this project

- Swift
  - [llm.swift](https://github.com/otabuzzman/llm.swift) by @[otabuzzman](https://github.com/otabuzzman): a Swift port of this project

- Zig
  - [llm.zig](https://github.com/Saimirbaci/llm.zig) by @[saimirbaci](https://github.com/Saimirbaci): a Zig port of this project

## discussions

Ways of organizing development:

- Experiencing a concrete issue with the repo? Use [Issues](https://github.com/karpathy/llm.c/issues).
- Have some code to contribute? Open a [PR](https://github.com/karpathy/llm.c/pulls)
- Chat about the repo, ask questions, etc.? Look at [Discussions](https://github.com/karpathy/llm.c/discussions).
- Something faster? I created a new `#llmc` channel on my [Zero to Hero Discord channel](https://discord.gg/3zy8kqD9Cp).

## license

MIT

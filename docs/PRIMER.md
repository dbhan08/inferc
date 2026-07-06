# Preface: what this is and who it's for

This document teaches the entire `inferc` project from the ground up. Here is exactly what
I assume about you, so nothing lands unfairly:

- **You can program** and you remember **basic math**: vectors and matrices, summation
  ($\sum$) notation, and what a derivative is. That's it.
- **I assume you know nothing** about machine-learning systems, how neural networks run on
  a computer, quantization, CPUs, or matrix coprocessors. Every one of those ideas is built
  up here from scratch, intuition first, with the formulas introduced only after the
  picture is clear. When a piece of math is a step beyond "basic" (the trace of a matrix,
  an outer product, a Hessian), I stop and define it.

By the end you will understand how a language model actually runs on a laptop, how `inferc`
compiles and executes one, what quantization is and the mathematics behind the best
methods, how a modern CPU secretly contains a matrix engine and how we program it, and the
two research results this project produced.

Read it front to back the first time — the nine parts are ordered so each uses only ideas
already introduced. A glossary and a repo map are at the end.

- **Part I** — the big picture: the problem, and what `inferc` is.
- **Part II** — how neural networks run (tensors, graphs, transformers, the two speed
  regimes).
- **Part III** — the `inferc` compiler and runtime.
- **Part IV** — the math of the hot code (matrix multiply, the roofline, attention).
- **Part V** — the hardware (CPUs, SIMD, and the Apple AMX matrix engine).
- **Part VI** — quantization and its mathematics (the mathematical heart).
- **Part VII** — the two research contributions.
- **Part VIII** — how we measure, and matching the field's standards.
- **Part IX** — a real bug we found, as a lesson in method.

---

# Part I — The Big Picture

## 1. The problem: using a trained network, fast

A neural network lives two lives. In **training**, you show it lots of data and slowly
tune its millions or billions of internal numbers (its **weights**) until it's good at a
task. Training is expensive and done once, usually on big GPUs. In **inference**, the
weights are frozen and you just *use* the network: feed an input in, read an answer out.
Every time a chatbot replies to you, that's inference.

This project is about **inference on CPUs** — the ordinary processors in laptops and
phones — specifically Apple's M1. Why CPUs, when GPUs are faster? Because that's what most
computers *are*. Running a language model directly on your laptop means your data never
leaves the device (privacy), there's no network delay, and there's no server bill. The
catch: CPUs are slower and have less memory, so you have to be clever. This whole project
is a study in that cleverness.

Two questions drive everything:

1. **Software.** Given a trained network, how do we run it on a CPU as fast as possible?
   (Parts III–IV.)
2. **Hardware.** The M1 chip hides an undocumented matrix-multiplying coprocessor called
   **AMX**. Can we program it directly and beat Apple's own libraries? (Parts V, VII.)

## 2. What `inferc` is

`inferc` ("infer, in C++") is two things in one repository:

- **An inference engine and optimizing compiler.** You hand it a trained network saved in
  the standard **ONNX** file format; it reads the file, builds its own internal picture of
  the network, cleans that picture up (mostly by merging many small operations into a few
  big ones — we'll see why that's fast), and then runs it with hand-written CPU code. It's
  small, readable C++, and it's checked to produce the same answers as the industry-standard
  engine (ONNX Runtime) down to floating-point rounding.
- **A hardware research vehicle.** A pile of small experiments and kernels that
  reverse-engineer and exploit the M1's AMX matrix engine. These produced two research
  papers, which are the real intellectual payoff.

The one idea to carry through everything below:

> On a modern CPU, the speed of neural-network inference comes down to a handful of big
> **matrix multiplications**, and how well you feed the hardware that does them.

Every part of this project — the compiler's merging passes, the quantization math, the AMX
kernels — exists to make those matrix multiplications cheaper, faster, or lower-energy.

---

# Part II — How Neural Networks Run

## 3. Tensors, operations, and the graph

Three words unlock everything.

A **tensor** is just an array of numbers with a shape. A single number is a 0-dimensional
tensor; a list of numbers is 1-D (a **vector**); a grid is 2-D (a **matrix**); a stack of
grids is 3-D; and so on. In a neural network *everything* is a tensor: the input, the
learned weights, and every partial result in between. Example: 128 words, each described
by 2048 numbers, is a tensor of shape $[128, 2048]$ — a grid with 128 rows and 2048
columns.

An **operation** (or "op") is one step of computation that takes tensors in and produces
tensors out: multiply two matrices, add two tensors, apply a softmax, and so on. A whole
network is built from a few dozen kinds of op.

A **computational graph** is the network drawn as a flowchart: boxes are operations,
arrows are tensors moving between them. If op A's output feeds op B, there's an arrow
A → B. Data only flows forward (the graph has no loops), so you can list the boxes in an
order where every box comes after the ones feeding it — running the network just means
walking that list and doing each op. (Computer scientists call such an ordering a
**topological order**; you don't need the term beyond "an order where inputs come first.")

One design detail in `inferc` matters later: in its graph, each op refers to its tensors
**by name** (a string), not by a direct pointer. There's one big table mapping names to
tensors. This makes *rewiring* the graph trivial — to reconnect two ops you just make a new
op whose input names match the outputs you want — which the optimization passes (Part III)
lean on heavily.

Finally, tensors come in two kinds:
- **Weights** (also called *constants* or *initializers*): fixed numbers baked into the
  model file, frozen during inference.
- **Activations**: tensors computed on the fly as data flows through.

## 4. ONNX: a common file format

Different training tools (PyTorch, TensorFlow) save models in their own formats. **ONNX**
(Open Neural Network eXchange) is a shared format they can all export to — a file that
lists the graph's operations, their settings, the weight tensors, and the input/output
shapes. `inferc` reads ONNX so it can run a model no matter what trained it. (The file
itself is stored as a *protobuf*, which is just a compact, structured binary format — think
"a strict, machine-readable form of JSON." You don't need to know more than that.)

## 5. What a transformer is (the kind of network we run)

Modern language models are **transformers**. You don't need the full theory, but you need
the shapes and the intuition, because the shapes are what the hardware sees.

**Step 1: words become vectors.** A transformer reads a sequence of tokens (words or
word-pieces). Each token starts as an integer ID. That ID is used to look up a row in a big
table called the **embedding table**, turning each token into a vector of length $H$ (the
"hidden size," e.g. 2048). So a 128-token input becomes a $[128, 2048]$ grid of numbers.
Intuition: the embedding is the model's learned "meaning coordinates" for each word.

**Step 2: a stack of identical layers refines those vectors.** Each **layer** has two
sub-parts.

*(a) Self-attention — letting words look at each other.* On its own, each word's vector
knows nothing about the sentence around it. Attention fixes that: it lets every word gather
information from the other words. Mechanically, from the input grid $X$ it first makes three
new grids by multiplying $X$ by three learned weight matrices — this is called a
**projection** (a projection is just "multiply your data by a weight matrix to get a new,
task-specific view of it"):
$$Q = XW_Q \quad(\text{"queries"}),\qquad K = XW_K \quad(\text{"keys"}),\qquad V = XW_V \quad(\text{"values"}).$$
Think of it as a lookup: each word's **query** asks "who is relevant to me?", each word's
**key** advertises "here's what I offer," and the **value** is the information actually
passed along. To find how much word $i$ should attend to word $j$, you compare $i$'s query
with $j$'s key by taking their dot product; doing this for all pairs is one matrix multiply,
$QK^\top$ (the $\top$ means *transpose* — flip rows and columns — so that the multiply lines
each query up against every key). Those raw comparison scores are turned into
attention *weights* that sum to 1 by a **softmax** (Section 16 explains softmax), and each
word's output is the weighted average of the values:
$$\text{Attention}(Q,K,V)=\text{softmax}\!\left(\frac{QK^\top}{\sqrt d}\right)V.$$
(The $\sqrt d$ just keeps the numbers from getting too large; $d$ is a size constant.)

*(b) Feed-forward network (FFN) — thinking about each word.* After attention mixes context
in, a small two-step network processes each word's vector on its own: expand it from size
$H$ to a bigger size (say $4H$), apply a smooth nonlinear "squish" function called **GELU**,
then shrink it back to $H$:
$$\text{FFN}(x)=\text{GELU}(xW_1+b_1)\,W_2+b_2.$$
(The $W$'s are weight matrices; the $b$'s are added bias vectors.)

Around each sub-part are a normalization step (**LayerNorm**, which rescales each word
vector to a stable range) and a "residual" addition (add the input back to the output, which
helps training). Stack $L$ of these layers (dozens, in a real model), then a final
projection to the size of the vocabulary, and a softmax, to get the probability of each
possible next word.

**The one thing to remember:** every heavy step here — the Q/K/V projections, the two
matrix multiplies inside attention, both FFN layers, the final vocabulary projection — is a
**matrix multiplication**. A transformer forward pass is, to first approximation, a
sequence of big matrix multiplies with cheap "glue" (softmax, LayerNorm, GELU, additions) in
between. That's why the rest of this document is obsessed with matrix multiply.

## 6. Two speed regimes: prefill vs decode

Generating text has two phases, and they stress the hardware in opposite ways. This split
runs through the whole project, so it's worth getting straight now.

- **Prefill** — reading the prompt. You feed the entire prompt in at once, say 128 tokens.
  Every matrix multiply has a "tall" dimension of 128 (many words handled together). These
  are big matrix-times-**matrix** products. Lots of arithmetic happens for each weight you
  load from memory, so the bottleneck is **doing the math** — it's *compute-bound*. This is
  where a matrix engine shines.
- **Decode** — writing the reply, one word at a time. Each new word depends on the
  previous, so you process a **single** token per step. Now the matrix multiplies have a
  tall dimension of 1 — they're matrix-times-**vector** products. Very little math happens
  per weight loaded, so the bottleneck is **moving the weights from memory** — it's
  *memory-bound*. A matrix engine can't help; the simpler vector unit, which just streams
  weights with minimal fuss, wins.

Hold onto this: **prefill is compute-bound (matrix-engine territory); decode is
memory-bound (vector-unit territory).** Part IV makes it precise.

---

# Part III — The `inferc` Compiler and Runtime

Now the actual engine. The `src/` folder is a small compiler with the classic three-part
shape: a **front end** that reads the input, a **middle** that cleans up an internal
representation, and a **back end** that runs it.

## 7. The pipeline at a glance

$$\text{ONNX file} \to \underbrace{\text{parse}}_{\text{frontend}} \to \underbrace{\text{internal graph (IR)}}_{} \to \underbrace{\text{figure out shapes}}_{} \to \underbrace{\text{optimize}}_{\text{merge ops}} \to \underbrace{\text{run}}_{\text{kernels}} \to \underbrace{\text{measure}}_{\text{profiler}}$$

## 8. Front end: read the file, build an internal graph

`src/frontend/onnx_loader.cc` opens the ONNX file and decodes it. Then
`src/frontend/onnx_to_ir.cc` walks that and builds `inferc`'s own internal graph — its
**IR** ("intermediate representation," meaning *the compiler's private, simplified copy of
the network*). It copies the weight tensors, records the inputs and outputs, and copies each
operation.

Why make a separate internal copy instead of using the ONNX objects directly? Because the
ONNX format is verbose and awkward to modify. A clean, minimal internal graph — a list of
ops plus one name-to-tensor table — is far easier to analyze and rewrite. Decoupling from
the input format is the first thing essentially every compiler does.

## 9. Figuring out the shapes

Before you can run or optimize anything, you need to know the shape (dimensions) and number
type of every tensor. `src/ir/shape_inference.cc` does this in **one forward pass**: because
the ops are already in an order where inputs come first, by the time you reach an op you
already know its inputs' shapes, so one sweep is enough. Each kind of op has a rule — a
matrix multiply of a $[M,K]$ by a $[K,N]$ gives $[M,N]$; an addition matches its inputs'
shapes; and so on.

A neat trick: some ops don't just infer a shape, they compute their actual *values* right
now, at compile time. If a reshape's target size is itself computed from the (already known)
size of another tensor, you can just do that little computation immediately and bake the
result in. This is called **constant folding** — doing at compile time any work that doesn't
depend on the actual input data — and it lets known information ripple deep into the graph.

## 10. Optimization: merging operations ("fusion")

This is where the compiler earns its keep. The passes live in `src/ir/passes/`. There are
two kinds.

- **Recognizers** spot a cluster of small ops that together implement a known mathematical
  function and replace the whole cluster with one custom op. For instance, a LayerNorm is
  exported by PyTorch as ~9 separate ops (subtract the mean, square, average, add a small
  number, square-root, divide, multiply, add...). `RecognizeLayerNorm` collapses all nine
  into a single `FusedLayerNorm`. Likewise `RecognizeAttention` collapses the entire
  attention pattern into one `FusedAttention`, and `RecognizeGelu` collapses the GELU chain.
- **Fusers** merge already-recognized neighbors — e.g. `FuseMatMulAddGelu` turns
  "matrix multiply → add bias → GELU" into a single fused op.

**Why bother merging?** Every op reads its inputs from memory and writes its output back.
A chain of five small ops makes five separate trips over the data — five times the memory
traffic. Merge them into one op and the data is read once, all five computations are applied
while it sits in fast registers/cache, and it's written once. Since this "glue" work is
memory-bound (Section 6), cutting the number of memory trips is a big win. Merging also lets
the fused op run one tight, optimized loop for the whole thing.

**How the passes stay correct.** Each pass scans for an "anchor" op, walks the arrows
checking that the neighboring op types and the tell-tale constants match (a LayerNorm's tiny
epsilon, attention's $1/\sqrt d$ factor), and — crucially — that each intermediate tensor is
used *only* by the next op in the pattern. If something else also reads that intermediate,
you can't safely fold it away. Only after collecting all the matches does it rebuild the op
list. There's also a constant-folding pass: if a transpose is applied to a fixed weight, do
the transpose once now and store the result.

## 11. Running the graph

`src/runtime/executor.cc` executes the cleaned-up graph. Up front it converts all the
weight bytes into ready-to-use tensors once. Then, each time you run the model, it:

1. builds a "tape" — a table mapping each tensor name to its current value — seeded with the
   inputs and the weights;
2. walks the ops in order; for each op it looks at the op type and calls the matching
   **kernel** (the concrete code that computes that op), reading inputs off the tape and
   writing outputs back;
3. at the end, reads out the model's declared outputs by name.

Ops run one at a time (serial), but the *expensive* ops (a big matrix multiply) split their
work across CPU cores internally. There's no fancy scheduler — the order from the graph is
the schedule.

## 12. Two memory tricks worth knowing

The engine keeps things fast with two deliberate choices that teach you where CPU time
actually goes:

- **Don't zero memory you're about to overwrite.** Allocating a tensor and filling it with
  zeros costs a full sweep over the buffer. Many kernels *completely overwrite* their output,
  so zeroing first is wasted work. The engine has an "allocate without zeroing" path
  (`Uninit`) for those, and only zeros (`Zeros`) when a kernel *adds into* its output.
- **Share memory for "view" ops.** Reshape, transpose, and slice often don't need to move
  any numbers — they're just a different *view* of the same bytes. The runtime tensor holds
  its storage in a shared, reference-counted buffer, so a view op can share the underlying
  data cheaply and only make a real copy when a kernel truly needs the numbers laid out
  contiguously.

## 13. Measuring: the profiler

`src/profiler/profiler.cc` is an optional stopwatch wired into the executor (it costs
nothing when off). It records, per op, the wall-clock time and peak memory, and writes a JSON
file. The Python benchmark that runs the reference engine (ONNX Runtime) writes the *same*
JSON format, so `inferc compare` can line the two up op-by-op. This is how every performance
claim in the project is made honestly — measured, not guessed.

The command-line tool (`src/cli/main.cpp`) exposes it all: `inspect` a model, `run` it,
`optimize` it, `compare` two profiles, `bench` against ONNX Runtime, `decode` (generate text
with a cache of past keys/values), and `amx-probe` (the hardware microbenchmark of Part V).

---

# Part IV — The Math of the Hot Code

## 14. Matrix multiply (GEMM): the beating heart

The single most important computation is **matrix multiply**, written $C = A\,B$, where
$A$ is $M\times K$, $B$ is $K\times N$, and the result $C$ is $M\times N$. Its definition is
the familiar "row times column":
$$C[i][j] = \sum_{k=0}^{K-1} A[i][k]\,B[k][j].$$
That's a triple loop ($i$, $j$, $k$) doing $M\cdot N\cdot K$ multiply-then-add operations. For
one transformer FFN with $M=128$, $K=2048$, $N=8192$, that's about 2 **billion**
multiply-adds — per layer, and there are dozens of layers. This op *is* the workload; its
common name is **GEMM** (GEneral Matrix-to-Matrix multiply). When the tall side $M=1$ (decode),
it's a matrix-times-vector, called **GEMV**.

There are two ways to organize the loops, and the difference is everything for hardware:

- **Inner-product form (dot products).** Fix one output cell $(i,j)$ and run the $k$ loop to
  build up its dot product, then move to the next cell. This is the natural way to write it,
  and it's how an ordinary vector unit does it.
- **Outer-product form.** Put the $k$ loop *outside*. For a single fixed $k$, you have one
  column of $A$ (the numbers $A[0][k], A[1][k], \dots$, a length-$M$ vector) and one row of $B$
  (a length-$N$ vector). An **outer product** of a length-$M$ column and a length-$N$ row is
  the $M\times N$ grid whose $(i,j)$ entry is (column entry $i$) × (row entry $j$). GEMM is
  just the sum of $K$ such grids:
  $$C = \sum_{k=0}^{K-1} \big(\text{column } k \text{ of } A\big)\big(\text{row } k \text{ of } B\big).$$
  (If "outer product" is new: a $2$-vector times a $2$-vector gives a $2\times2$ grid,
  $\begin{psmallmatrix}u_0\\u_1\end{psmallmatrix}\begin{psmallmatrix}v_0&v_1\end{psmallmatrix}
  =\begin{psmallmatrix}u_0v_0&u_0v_1\\u_1v_0&u_1v_1\end{psmallmatrix}$. That's it.)

A **matrix engine** (Part V) does the outer-product form: one instruction takes a length-16
chunk of a column of $A$ and a length-16 chunk of a row of $B$ and updates a whole
$16\times16$ block of $C$ — 256 multiply-adds in one shot. Both forms compute the exact same
answer; they differ only in loop order, and therefore in which hardware is efficient.

## 15. When are you compute-bound vs memory-bound? The roofline

Why does prefill favor the matrix engine and decode favor the vector unit? One number
answers it: **arithmetic intensity**, the ratio of math done to bytes moved,
$$I = \frac{\text{arithmetic operations}}{\text{bytes read from memory}}\quad[\text{ops per byte}].$$
A machine has two ceilings: a top compute rate $P$ (operations/second) and a top memory
bandwidth $B$ (bytes/second). The most you can achieve is
$$\text{speed} \le \min\big(P,\; I\times B\big).$$
Picture it as a roof: for low intensity you're on the slanted part ($I\times B$, limited by
memory); past a "knee" you hit the flat part ($P$, limited by compute). Which side you're on
decides your fate:

- **Prefill GEMM** ($M=128$): each weight you load is reused across all 128 tokens, so you
  do a lot of math per byte — high intensity, past the knee — **compute-bound**. Bring a
  matrix engine.
- **Decode GEMV** ($M=1$): each weight is used exactly once (one token), so intensity is
  tiny — below the M1's knee — **memory-bound**. You're limited by how fast weights stream
  from memory, and no compute engine helps. This is exactly *why* quantization (fewer bytes
  per weight, Part VI) is the lever that speeds up decode.

## 16. Softmax and attention, carefully

Attention (Section 5) is two matrix multiplies with a **softmax** between them. Softmax turns
a list of raw scores $s$ into positive weights that sum to 1 (a probability distribution):
$$\text{softmax}(s)_i = \frac{e^{s_i}}{\sum_j e^{s_j}}.$$
Done naively this blows up: $e^{s_i}$ is astronomically large if $s_i$ is, say, 100. The
standard fix uses a simple fact — subtracting the same constant $c$ from every score leaves
the result unchanged (the $e^{-c}$ factors cancel top and bottom). Choose $c=\max_j s_j$, and
now the biggest exponent is $e^0 = 1$: no overflow. Every softmax in this codebase does this
max-subtraction.

`inferc`'s attention kernel is written "flash-attention style," meaning it never builds the
big intermediate score grid for the whole input at once. It processes one attention head at a
time into a small scratch buffer, does the stable softmax in place, and multiplies by the
values straight into the output — the same "minimize trips over memory" philosophy as the
fusion passes, applied inside one op.

The other glue ops — LayerNorm (normalize each word vector), GELU (the smooth squish
nonlinearity), and elementwise add/multiply — are all memory-bound, so the kernels lean on
Apple's vectorized math routines and touch memory as few times as possible. None of them
dominate runtime; they exist so the matrix multiplies, which *do* dominate, are surrounded by
as little overhead as possible.

---

# Part V — The Hardware: CPUs, SIMD, and the AMX Matrix Engine

To beat Apple's libraries you have to know what's physically inside the chip.

## 17. Cores and SIMD (doing several multiplies at once)

A CPU **core** runs instructions one after another. To get more done per instruction, cores
have **SIMD** units — "Single Instruction, Multiple Data." The idea: a special wide register
holds several numbers, and one instruction operates on all of them at once. ARM's version is
called **NEON**: one register holds 4 floats, and a single NEON multiply-add does 4
multiply-adds together. To multiply matrices with NEON you use the dot-product form
(Section 14) and issue lots of these 4-wide instructions. Two facts matter later: NEON does a
*1-dimensional* chunk of work per instruction, and **each core has its own NEON unit**, so
NEON throughput grows with the number of cores (8 cores → 8 NEON units).

Analogy: a scalar core is one worker doing one multiplication at a time; SIMD is one worker
with a 4-slot jig, doing four identical multiplications in one motion.

## 18. A matrix engine, and Apple AMX

A **matrix engine** goes one dimension further than SIMD: a single instruction takes two
vectors and updates a whole 2-D *grid* of running totals — an outer product (Section 14).
That's quadratically more arithmetic per instruction, which is why matrix engines dominate
matrix multiply. In the jig analogy: instead of a 4-slot line, it's a $16\times16$ **stamp**
that presses out 256 multiply-adds at once.

**Apple AMX** ("Apple Matrix eXtension") sits inside the M1/M2/M3. The catch: it's
**undocumented** — Apple never published how to use it. It's reachable only through a
community-reverse-engineered instruction encoding, which this repo includes as a tiny 34-line
header (`third_party/amx/aarch64.h`) that emits each AMX instruction as a raw machine-code
word. Its anatomy:

- **Two operand register files, X and Y**: 8 registers each, holding the two matrices'
  chunks.
- **An accumulator, Z**: a $64\times64$-byte grid where results pile up.
- **It's shared per *cluster*, not per core.** The M1 has two clusters of cores (four fast
  "performance" cores and four "efficiency" cores), and there is **one AMX block per
  cluster** — so only ~2 AMX blocks exist on the entire chip, no matter how many threads you
  launch. Contrast NEON, which scales to all 8 cores. This single fact shapes the whole
  performance and energy story below, so tuck it away.

## 19. Outer products on AMX — a worked example

One AMX floating-point matrix instruction computes a $16\times16$ outer product:
$$Z[i][j] \mathrel{+}= X[i]\times Y[j],\qquad 0\le i,j<16,$$
where $X$ holds 16 numbers (a chunk of a column of $A$) and $Y$ holds 16 numbers (a chunk of
a row of $B$). That's $16\times16=256$ multiply-adds in one instruction. A full result block is
built by looping over $k$ and letting the totals accumulate in $Z$, then storing $Z$ out.

Tiny example with $K=2$ (imagine a 2-wide stamp). Let
$A=\begin{psmallmatrix}a_{00}&a_{01}\\a_{10}&a_{11}\end{psmallmatrix}$ and
$B=\begin{psmallmatrix}b_{00}&b_{01}\\b_{10}&b_{11}\end{psmallmatrix}$.
- $k=0$: outer product of column $[a_{00},a_{10}]$ and row $[b_{00},b_{01}]$ →
  $\begin{psmallmatrix}a_{00}b_{00}&a_{00}b_{01}\\a_{10}b_{00}&a_{10}b_{01}\end{psmallmatrix}$.
- $k=1$: outer product of column $[a_{01},a_{11}]$ and row $[b_{10},b_{11}]$ →
  $\begin{psmallmatrix}a_{01}b_{10}&a_{01}b_{11}\\a_{11}b_{10}&a_{11}b_{11}\end{psmallmatrix}$.

Add the two grids: the top-left cell is $a_{00}b_{00}+a_{01}b_{10}$, which is exactly
$\sum_k A[0][k]B[k][0]$ — the right answer. The engine accumulated both outer products in $Z$
automatically.

Different number types give different stamp sizes: 32-bit floats give $16\times16$; 8-bit
integers pack 64 values per register for a denser $64\times16$; 16-bit floats give
$32\times32$. More values per register means more multiply-adds per instruction — a lever we
use later.

*(One subtlety, for intuition: if every instruction writes the same $Z$ grid, each must wait
for the previous to finish — a serial chain that idles the engine. The kernels fix this by
using four separate $Z$ grids so four instructions are "in flight" at once. This "keep
independent work flowing so the unit never waits" idea returns in Part IX.)*

## 20. The load-issue ceiling — Paper 1's core finding

Here's the finding that shapes everything. You'd expect that with 256 multiply-adds per
instruction, AMX is blindingly fast. It is — *if you never have to load new operands*. A
microbenchmark that fires a pure stream of multiply-adds with no loads hits about **1525
billion ops/second** on one thread. But a real matrix multiply must constantly load fresh
chunks of $A$ and $B$. The instant even *one* operand-load is mixed into the multiply stream,
single-thread throughput collapses to a **610–680 billion-ops/second band** — and stays
there no matter how you rearrange the work:

| single-thread inner loop | loads : multiplies | billion ops/s |
|---|---|---|
| multiplies only (no loads) | 0 : 4 | ~1525 |
| + 1 load | 1 : 4 | ~677 |
| + 3 loads (the shape we ship) | 3 : 4 | ~643 |

**What it means:** on the M1's AMX, operand *loads* and matrix *multiplies* compete for the
same single "issue slot" — the engine can't start a load and a multiply in the same cycle. So
it starves, not for lack of math power but for lack of a way to feed itself. This is the
**load-issue ceiling**, and no single-threaded trick escapes it.

But **multiple threads hide it.** With several threads sharing one AMX block, one thread's
loads slip into the gaps another thread's multiplies leave behind. Loaded throughput climbs
from ~673 (1 thread) to ~1483 (8 threads) billion ops/s — a 2.2× gain — while the
pure-compute rate barely moves. That proves the win comes from *hiding loads across threads*,
not from computing faster. Paper 1 turns this understanding into a kernel that beats Apple's
library (Part VII).

## 21. Why prefill wins and decode loses — the energy story

Because there's only one AMX block per cluster, a single AMX thread does the work of the
whole vector array — while using far fewer cores, and therefore far less power. That's the
real payoff. On a representative quantized matrix multiply (Part VI):

| engine | active cores | billion ops/s | power (W) | ops/s per watt |
|---|---|---|---|---|
| AMX | 4 | 845 | 6.2 | **136** |
| NEON (SIMD) | 8 | 549 | 18.4 | **30** |

Similar throughput, but AMX draws **one-third the power** and delivers **~4.6× the work per
watt** — because it hits that throughput on a quarter of the cores. For a battery-powered,
heat-limited laptop, energy per token is what actually matters, and the matrix engine wins
decisively.

For **decode**, though, the job is memory-bound (Section 15): a good NEON kernel already
saturates the M1's memory bandwidth, so AMX — which has no edge once you're bandwidth-limited
— actually loses. Hence the project's rule of thumb: **AMX for prefill, NEON for decode.**

---

# Part VI — Quantization and Its Mathematics

This is the mathematical heart of the project and the subject of Paper 2. Take it slowly; the
GPTQ part (Section 26) is the hardest, and it's also where a real bug lived (Part IX).

## 22. Why quantize at all

A weight stored as a 32-bit float takes 4 bytes. A 7-billion-parameter model is then 28
gigabytes — too big for a laptop's memory, and (recall Section 15) decode is *memory-bound*,
so its speed is set by how fast you stream those bytes. **Quantization** stores each weight in
*fewer bits* — 4 bits is standard, an 8× shrink from 32. That lets the model fit in memory and
speeds up memory-bound decode almost proportionally. The whole challenge is: how do you throw
away most of the bits per weight while barely hurting the model's answers?

The key term is **bitwidth**, $b$: a $b$-bit weight can be one of $2^b$ values. "4-bit" means
16 possible values per weight. "At equal bitwidth" means comparing two schemes that use the
same memory, so the only question is *which* 16 values you allow and *how* you choose them.

## 23. Uniform (evenly-spaced) quantization

The simplest scheme spaces the allowed values **evenly**. Pick a step size $s$ and a starting
offset $z$; store each weight as a small integer $q\in\{0,1,\dots,2^b-1\}$, and recover an
approximate weight by
$$\hat w = s\,q + z.$$
To go the other way (encode a real weight $w$): round $(w-z)/s$ to the nearest integer and
clamp it to the allowed range. If you set $z=\min$ and $s=(\max-\min)/(2^b-1)$ over a block of
weights, the 16 levels span exactly from the block's smallest to largest value.

Why is this the CPU default despite being crude? Because *decoding is one multiply and one
add* — trivial for any vector unit. But it wastes resolution. Neural-network weights are
**bell-shaped**: lots of them cluster near zero, few are out in the tails. Evenly-spaced levels
spend as many of their 16 values out in the sparse tails as in the crowded middle — where most
of the weights, and therefore most of the error, actually are.

**Per-group scales.** One step size for a whole 4096-wide row of weights is too coarse: a
single unusually large weight stretches $\max$ and wastes the grid on empty range. So a row is
chopped into **groups** (here, 64 consecutive weights) and each group gets its own $s$ and $z$.
Cost: about half an extra bit per weight to store the scales. Benefit: much less error. Smaller
groups = more accurate = more overhead. This "group size" is a standard knob (we use 64; the
field often uses 128).

## 24. Non-uniform (codebook) quantization: NF4 and k-means

A **codebook** drops the even-spacing rule. You keep a little table of $2^b$ arbitrary real
values — the **codebook** $C = [c_0, c_1, \dots]$ — and store each weight as the **index** of
its closest table entry:
$$\text{encode: } q_i=\arg\min_k |w_i-c_k|,\qquad \text{decode: } \hat w_i=c_{q_i}.$$
($\arg\min_k$ just means "the index $k$ that makes the following smallest.") Now you can put
levels *where the weights actually are* — dense near zero, sparse in the tails — so you get
less error with the same 16 values. The price: decoding is now a **table lookup** (a "gather"):
turn an index into a value. On a vector unit that lookup is a separate shuffle instruction
*before* the multiply — extra work. That's exactly why CPUs avoided codebooks, and exactly the
cost Paper 2's free hardware gather erases (Part VII).

Two ways to pick the levels:

**NF4 (NormalFloat-4).** A *fixed* codebook whose 16 levels are the quantiles of a standard
bell curve — the mathematically right spacing *if* weights are perfectly bell-shaped. The
actual values (note how they bunch up near 0 and spread out toward $\pm1$):
$$[-1,\,-0.70,\,-0.53,\,-0.39,\,-0.28,\,-0.18,\,-0.09,\,0,\,0.08,\,0.16,\,0.25,\,0.34,\,0.44,\,0.56,\,0.72,\,1].$$
To fit a real group you scale these by the group's largest magnitude, $s=\max_i|w_i|$, so the
levels stretch to cover that group. This is the standard, off-the-shelf non-uniform choice.

**k-means codebook.** Instead of fixed levels, *fit* the 16 levels to each group's actual
numbers, minimizing squared error. The classic method (Lloyd's algorithm) repeats two steps
until settled: **(1) assign** each weight to its nearest current level; **(2) update** each
level to the average of the weights assigned to it. Each round provably lowers the total
squared error, converging to the best possible 16 levels *for that group's weights*.

Here's a conceptual trap that Part IX springs: k-means minimizes **weight** error better than
NF4, yet in the full model it barely helps. Why? Because the network doesn't care about weight
error — it cares about **output** error. Minimizing "how far are the stored weights from the
originals" is the wrong target. Fixing that needs the next method.

## 25. GPTQ: fixing the *output*, not the weights, via error feedback

**GPTQ** is the method that reaches top-tier 4-bit accuracy, and it does so not with fancier
levels but with **error feedback**: quantize the weights one column at a time, and every time
rounding a column introduces an error, *nudge the columns you haven't done yet* to cancel that
error's effect on the layer's output. Here's the whole idea, built up gently.

**A quick notation reminder.** The transpose $M^\top$ flips rows and columns. The **trace**
$\operatorname{tr}(M)$ is just the sum of a square matrix's diagonal entries. The **Frobenius
norm** $\|M\|_F^2$ is the sum of the squares of all of $M$'s entries — a plain "total squared
size." That's all the extra machinery we need.

**The right objective.** A layer computes (activations) × (weights): outputs $= XW^\top$,
where $X$ is a stack of example inputs (from a small **calibration** set of real text) and $W$
is the weight matrix. We want the quantized weights $Q$ to keep the *output* close to the
original, so we minimize the total squared change in the output:
$$\big\|X(W-Q)^\top\big\|_F^2 \;=\; \operatorname{tr}\!\big((W-Q)\,\underbrace{X^\top X}_{H}\,(W-Q)^\top\big).$$
(The equality is a standard algebra identity; you can take it on faith.) The matrix in the
middle,
$$\boxed{H = X^\top X},$$
is called the **Hessian**. In general a Hessian is the matrix of a function's second
derivatives — it measures curvature, i.e. how sharply the objective rises as you move the
weights — and for this quadratic objective it works out to exactly $X^\top X$. Its meaning is
very concrete: $H$ is (proportional to) the covariance of the layer's inputs. A big diagonal
entry $H_{jj}$ means input feature $j$ is usually active, so weight $j$ matters a lot; a big
off-diagonal $H_{jk}$ means features $j$ and $k$ tend to fire together, so their errors can be
made to cancel. Minimizing error "weighted by $H$" instead of plain weight-distance is exactly
what makes GPTQ care about the *output*.

**Getting $H$.** You run a **calibration** set — a chunk of representative text — through the
model and, at each layer, add up $H = \sum_{\text{tokens}} x\,x^\top$ over the actual inputs $x$
that layer sees. That's why GPTQ is "data-aware": $H$ records which input directions the layer
really uses.

**The greedy fix.** Quantize columns left to right. When you round column $w_j$ to the grid,
you make an error $\delta_j = w_j - q_j$. The best first-order way to compensate — to keep the
output unchanged — is to spread that error into the remaining, not-yet-quantized columns,
weighted by the inverse Hessian $H^{-1}$:
$$W[:,\,j{+}1{:}] \;\mathrel{-}=\; \frac{\delta_j}{[H^{-1}]_{jj}}\,[H^{-1}]_{j,\,j{+}1:}.$$
Read it as: "I was forced to round column $j$ badly; the input statistics ($H^{-1}$) tell me
how to tweak the correlated later columns to undo the damage to the output." Columns already
done are frozen; only future ones absorb the error, so it's a single left-to-right sweep, and
it's provably the exact best correction under "each column must land on the grid."

**Two numerical safety nets** (in the code): **damping** — add a small amount to $H$'s
diagonal so it's safely invertible (a rounded-off $H$ can otherwise be non-invertible); and the
**Cholesky trick** — a standard, stable way to get the inverse-Hessian information without
forming a full inverse. Real implementations add two refinements: a **blocked** version (do the
bookkeeping 128 columns at a time for cache-friendliness — same math), and **activation
ordering** (quantize the most important columns first, while the full error budget is still
available to soak up their error).

**Why it wins.** GPTQ happily accepts large weight errors in directions the data never
exercises, and spends its precision on the directions the data *does* exercise. That's why
"NF4 + GPTQ" is the most accurate 4-bit method in our tables — beating k-means even though
k-means has smaller raw weight error. The moral: *error feedback, not a cleverer codebook, is
what reaches state-of-the-art 4-bit.*

**A caution you'll cash in at Part IX.** $H = X^\top X$ can only "see" as many independent
directions as you gave it calibration tokens. If you use too few tokens compared to the layer's
width, $H$ is **rank-deficient** — blind in some directions — and the damping term is all
that's holding its inverse together there. GPTQ then steers its error feedback along
directions that are pure noise, *overfits* the tiny calibration sample, and can make held-out
accuracy *worse*. Enough calibration data isn't a nicety; it's correctness.

## 26. Perplexity: measuring the damage

To score how much a quantizer hurt the model, we use **perplexity** on held-out text
(the WikiText-2 benchmark). A language model assigns a probability to each actual next word.
Perplexity is
$$\text{PPL} = \exp\!\Big(\tfrac1N\sum_{t=1}^N -\log p(x_t\mid x_{<t})\Big),$$
that is, $e$ raised to the average "surprise" ($-\log$ probability) per word. **Lower is
better**: a perplexity of 20 means the model is, on average, as unsure as if it were guessing
among 20 equally-likely words. We report the *percent increase* in perplexity over the
full-precision model — how much accuracy a quantizer costs.

---

# Part VII — The Two Research Contributions

The project produced two companion papers, both about the M1's AMX, split by precision.

## 27. Paper 1 — beating Apple's library at prefill matrix multiply (full precision)

**Claim:** a hand-written 32-bit matrix-multiply kernel can beat Apple's own tuned library
(Accelerate) at LLM prefill — **not by computing faster** (Section 20 showed the inner loop is
already pinned at the hardware's load-issue ceiling) but by two deployment tricks Apple's
general-purpose routine skips:

1. **Pre-pack the weights once.** To feed the AMX stamp, an operand must be rearranged into a
   special layout ("packing"). Apple's routine re-packs the weight matrix on *every* call. But
   in inference the weight is *constant* all session — so pack it **once** at load time and
   reuse it forever. This wins the shapes where the output is wide.
2. **Use both AMX blocks with fine-grained threading.** By splitting the output into many
   narrow strips instead of a few wide ones, the work spreads across all cores and — crucially
   — reaches the **second AMX block on the efficiency cluster** that Apple's routine leaves
   completely idle. This wins the shapes where the shared dimension is large.

Result: **bit-for-bit identical** to Apple's output, yet about **1.58× faster** on average
across the twelve prefill matrix shapes, and up to **1.44× faster end-to-end** when dropped
into the real `llama.cpp` runtime. The paper's rigor is its long list of things that *don't*
help — an honest map of dead ends that leaves strip width as the one knob that matters.

## 28. Paper 2 — the free codebook gather (4-bit) — the flagship

Recall the tension from Part VI: codebook (non-uniform) quantization is more accurate than
uniform at the same bitwidth, but was thought impractical on CPUs because the codebook lookup
is slow on the vector unit. Paper 2 shows **that penalty is only there because people used the
wrong part of the chip.**

**The mechanism.** Apple's AMX matrix instructions carry an undocumented **"indexed-load"
bit**. When it's set, one operand register is read not as data but as packed **4-bit
indices**, and the hardware **looks each one up in a 16-entry codebook held in another
register** *before* doing the outer product. So a single instruction does lookup +
dequantize + multiply-accumulate, fused together. Three properties were checked directly with
microbenchmarks:

- **Correct** — with codebook $[100,101,\dots]$ and indices $[0,1,\dots]$, it returns exactly
  $100,101,\dots$, bit for bit.
- **Free** — an indexed matrix instruction takes ~1.15 cycles, the *same* as a plain one. The
  lookup hardware sits in the operand-read path and costs nothing extra.
- **Specific** — only the *matrix* instructions do this; the plain vector multiply ignores the
  bit. (An early guess that the vector unit could do it was tested and *disproved* — a nice
  example of checking correctness before trusting a speed number, which is Part IX's theme.)

So a **non-uniform 4-bit matrix multiply costs the same as a plain float matrix multiply, with
the dequantization free and the result bit-exact.** As far as we know this is the first
single-instruction "look-up-and-multiply" shown on a shipping CPU matrix engine.

**The three payoffs:**

1. **Accuracy for free (better, in fact).** Because the kernel is bit-exact, the accuracy is
   entirely the quantizer's and can be measured in ordinary PyTorch. At 4-bit on WikiText-2
   (percent perplexity increase over full precision, lower is better):

   | 4-bit method | GPT-2 | TinyLlama-1.1B |
   |---|---|---|
   | uniform | +18.5% | +9.3% |
   | NF4 codebook | +11.2% | +5.4% |
   | **NF4 + GPTQ** | **+2.9%** | **+5.1%** |

   Non-uniform never loses to uniform at equal bitwidth — on GPT-2 it degrades 6.5× less.
   (The TinyLlama GPTQ number is the subject of Part IX.)

2. **Speed for free.** At small-batch prefill the codebook matrix multiply runs at uniform-4-bit
   speed — up to **4× faster than llama.cpp's optimized uniform NEON kernel** — because the
   lookup is free. It loses only at single-token decode, which is memory-bound and belongs to
   NEON.

3. **Energy — the real prize.** Reaching its speed on a quarter of the cores, the matrix
   engine uses about **one-third the power** and **4.6× the work-per-watt** of the vector path
   (Section 21's table).

**The framing.** On a plot of accuracy vs. energy, two *independent* free moves — put any
format on the matrix engine (⅓ the energy, same accuracy) and switch uniform→codebook at no
extra cost (better accuracy, same energy) — combine into a point that **beats** uniform-on-the-
vector-unit on *both* axes at once. Not a trade-off — a strict improvement. The broader lesson:
*whether a quantization format is practical is decided by whether the target hardware can do
the lookup cheaply, and CPU matrix engines already can.*

---

# Part VIII — How We Measure, and Matching the Field's Standards

A result is only as good as its measurement. Two things get measured: speed/energy (on the
M1, with microbenchmarks and a power meter — solid methodology) and **accuracy** (in PyTorch,
which is valid *because* the kernel is bit-exact, so the PyTorch model stands in perfectly for
the hardware). The accuracy *methodology*, assessed honestly, is currently **below** the
protocol the quantization research field uses — and to compare fairly against published
numbers, we must match it. The gap, axis by axis:

| Axis | What our harness does | Field standard | Fix |
|---|---|---|---|
| **Models** | GPT-2 (124M), TinyLlama-1.1B | Llama-2-7B/13B, Mistral-7B | Make Llama-2-7B the primary model |
| **Eval data** | WikiText-2 only | WikiText-2 **and** C4 | Add a C4 perplexity pass |
| **Sequence length** | 512 | 2048 | Evaluate at 2048 |
| **Calibration** | ~4096 tokens from WikiText-train | **~262k tokens from C4** | Use 128×2048 C4 segments |
| **Group size** | 64 | 128 | Also report group-128 |
| **Bit-widths** | 4-bit | 4-bit, also 3- and 2-bit | Add 3- and 2-bit |
| **Downstream tasks** | 3 tasks, 200 examples | 5–6 tasks, full sets | Run the full standard suite |

Two of these are not mere presentation choices — they're **correctness**:

- **Where the calibration text comes from.** Calibrating GPTQ on WikiText-*train* and then
  testing on WikiText-*test* is a subtle form of cheating (the two are the same kind of text);
  the field calibrates on a *different* corpus (C4) on purpose.
- **How much calibration text.** 4096 tokens is 30–60× below the standard, and — as Part IX
  shows — it's few enough that on real large layers the Hessian $H$ goes rank-deficient and
  GPTQ overfits. Fixing the calibration size fixes the leakage, the comparability, *and* the
  bug all at once.

The good news: the *quantizer math* in the repo is genuinely state-of-the-art shaped (real
blocked GPTQ with activation ordering, proper error feedback, NF4 and k-means codebooks). It's
the *evaluation* that needs to be brought up to protocol before any number is compared to
published work.

---

# Part IX — A Real Bug, as a Lesson in Method

The best way to end is with an actual research episode from this project, because it teaches
the *method*, not just the facts. It ties together GPTQ, the Hessian, calibration, and
perplexity.

**The symptom.** In the accuracy table (Section 28), GPTQ's error feedback helps GPT-2
enormously (NF4 +11.2% → NF4+GPTQ +2.9%) but barely helps TinyLlama-1.1B (+5.4% → +5.1%). Why
would the *same method* transform one model and do nothing for a bigger one? The easy story —
"bigger models are just more robust to quantization, so there's less to gain" — is *partly*
true, but it was hiding a bug.

**The discipline.** A first measurement is a hypothesis, not a conclusion; before writing
"the gain shrinks with scale," find out *why*. Recall from Section 25 that GPTQ's Hessian
$H = X^\top X$ can only see as many directions as you gave it calibration tokens. We used
4096. TinyLlama's FFN "down-projection" layer has an input width of **5632**. Since
$4096 < 5632$, that layer's Hessian is **rank-deficient** — literally blind in over a thousand
directions — so GPTQ's error feedback is steering along noise there.

**The decisive test.** Instead of an expensive full perplexity run, we measured GPTQ's *own*
objective — the output error it's supposed to minimize — on two *separate* batches of data: a
**training** batch (the calibration data it optimizes against) and a **held-out** batch (fresh
data). If GPTQ improves the training batch but *worsens* the held-out batch, that's the
signature of overfitting. Results (ratio = GPTQ's error ÷ plain-NF4's error; below 1 means
GPTQ helped, above 1 means it *hurt*):

| layer | with 4096-token calibration (held-out) | with 32k-token calibration (held-out) |
|---|---|---|
| down-proj (width 5632) | **1.71–1.99 — GPTQ HURT** | **0.58–0.86 — GPTQ helped** |
| gate-proj (width 2048) | 0.95–1.40 (nil to hurt) | 0.54–0.76 — helped |
| attention q-proj | 0.17–1.16 | 0.17–0.61 — helped |

On the calibration data GPTQ crushed the error everywhere (it *works*). On fresh data, with too
little calibration, it *hurt* the big FFN layers — which are the majority of the model's
weights — so overall GPTQ came out roughly even with plain NF4. Give it enough calibration
(32k tokens, comfortably more than 5632) and every layer flips to genuinely helping.
**Diagnosis confirmed in both directions: the weak TinyLlama result was a
calibration-too-small artifact — not a property of the model, the method, or the AMX kernel.**
GPT-2 escaped only because its layers ($\le3072$ wide) stayed full-rank even at 4096 tokens.

**The lessons, which generalize far past this project:**
1. **A surprising result is an artifact until proven otherwise** — in *either* direction; a
   suspiciously good number deserves as much suspicion as a disappointing one.
2. **Check correctness before believing a speed or quality number.** The held-out test
   separated two different claims the single number had blurred together: "GPTQ is working"
   (training error down) and "GPTQ is helping" (held-out error down).
3. **State the cause in the system's own terms.** The conclusion isn't "GPTQ is weak on
   TinyLlama"; it's "when calibration tokens (4096) < layer width (5632), the Hessian is
   rank-deficient, so error feedback overfits" — a cause you can act on. And the fix (much more
   calibration data, from C4) is exactly what the field's standard protocol (Part VIII) already
   demanded.

That's the whole project in miniature: measure carefully, distrust the first number, find the
mechanism, and let the mechanism tell you the fix.

---

# Appendix A — Glossary

- **Activation** — a tensor computed while the model runs (as opposed to a fixed weight).
- **AMX (Apple)** — Apple's undocumented per-cluster matrix engine; does $16\times16$
  multiply-add grids. (Unrelated to Intel's same-named unit.)
- **Arithmetic intensity** — math done per byte moved; decides compute- vs memory-bound.
- **Bit-exact** — identical to a reference answer down to the last bit.
- **Bitwidth ($b$)** — bits per weight; gives $2^b$ possible values.
- **Calibration set** — sample text run through the model to gather statistics (the Hessian)
  for GPTQ.
- **Codebook (non-uniform) quantization** — store weights as indices into a small table of
  learned values; more accurate than uniform, needs a lookup to decode.
- **Decode** — generating one token at a time; memory-bound; NEON's job.
- **Frobenius norm ($\|M\|_F^2$)** — sum of the squares of all a matrix's entries.
- **GELU** — the smooth nonlinear "squish" in a transformer's FFN.
- **GEMM / GEMV** — matrix-matrix / matrix-vector multiply.
- **GPTQ** — 4-bit quantization using Hessian-guided error feedback.
- **Group / per-group scale** — a separate scale for each small block of weights.
- **Hessian ($H = X^\top X$)** — curvature of GPTQ's objective; the input covariance; the
  "metric" GPTQ minimizes error in.
- **IR** — intermediate representation; the compiler's private copy of the network.
- **LayerNorm** — normalizes each token's vector to a stable range.
- **NF4** — a fixed 4-bit codebook whose levels are bell-curve quantiles.
- **ONNX** — the standard file format for saving neural networks.
- **Outer product** — a column times a row, giving a grid; matrix multiply is a sum of these.
- **Perplexity** — $e^{\text{average per-word surprise}}$; model quality, lower is better.
- **Prefill** — reading the whole prompt at once; compute-bound; AMX's job.
- **Quantization** — storing weights in fewer bits to save memory and bandwidth.
- **Rank-deficient** — a matrix that's "blind" in some directions (here, from too little
  calibration data); makes GPTQ overfit.
- **Roofline** — the rule speed $\le \min(\text{compute limit},\ \text{intensity}\times\text{bandwidth})$.
- **SIMD / NEON** — a per-core unit that does several identical operations at once; scales with
  core count.
- **Softmax** — turns raw scores into probabilities that sum to 1.
- **Tensor** — an array of numbers with a shape.
- **Trace ($\operatorname{tr} M$)** — sum of a square matrix's diagonal.
- **Transpose ($M^\top$)** — flip a matrix's rows and columns.
- **Uniform (affine) quantization** — evenly-spaced levels, decode $\hat w = sq + z$.
- **WikiText-2** — the standard text benchmark for measuring perplexity.

# Appendix B — Where to look in the repo

- **Compiler / engine:** `src/frontend` (read ONNX → build IR), `src/ir` (the graph, shape
  inference, `passes/` for op-merging), `src/runtime` (the executor + tensors),
  `src/kernels` (matrix multiply, attention, activations, the direct-AMX `amx_gemm.cc`),
  `src/profiler`, and `src/cli/main.cpp` (the command-line tool).
- **AMX research:** `third_party/amx/aarch64.h` (the instruction encodings); `bench/amx/*.cc`
  — the codebook-gather crux is `amx_matfp_indexed.cc` (it's correct),
  `amx_matfp_indexed_cost.cc` (it's free), `amx_codebook_gemm.cc` / `amx_e2e_codebook.cc`
  (the kernel), and `amx_energy_load.cc` (the power measurement).
- **Quantization + evaluation:** `bench/amx/codebook_perplexity.py` (uniform / NF4 / k-means /
  GPTQ + the Hessian + perplexity), `gptq_eval.py` (the full blocked, activation-ordered
  GPTQ), `ppl_fast.py` (the perplexity driver), and the Part IX diagnostics
  `diag_gptq_tinyllama.py` / `diag_gptq_fullrank.py`.
- **Papers:** `docs/PAPER_DRAFT.md` (Paper 1), `docs/PAPER2_NEURIPS_DRAFT.md` and
  `docs/PAPER2_DRAFT.md` (Paper 2), `docs/AMX_REPORT.md` (the hardware investigation).

# Appendix C — Building this document

```
bash docs/build_primer.sh      # renders docs/PRIMER.md → docs/PRIMER.pdf
```

Requires `pandoc` and `tectonic`. The script adds the title/table-of-contents front matter,
converts the math symbols to LaTeX, and compiles a single-column PDF.

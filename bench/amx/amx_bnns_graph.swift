// BNNS Graph head-to-head for the M1 AMX paper (macOS 15+ / Tahoe 26).
//
// BNNS Graph is Apple's auto-weight-repacking path -- a whole-graph AOT compiler,
// NOT a drop-in GEMM call like cblas_sgemm/BNNSMatMul. This times it on the same
// 12 fp32 LLM-prefill matmuls the paper's pre-packed kernel targets, so we can
// report the one comparison reviewers will ask for (the #1 reject risk, paper §5).
//
// FAIRNESS, matched to the paper's amortized pre-pack:
//   - the constant weight W is baked into each .mlpackage (gen_bnns_graph_models.py,
//     forced fp32 so this is bit-exact-precision with the paper, not BNNS Graph's
//     default fp16 downcast);
//   - Context(compileFromPath:) compiles + repacks W ONCE at load -> outside timing;
//   - only executeFunction is timed, median-of-trials, run 11x for the campaign.
//
// Build (needs full Xcode SDK for the BNNSGraph Swift overlay, macOS 15+):
//   swiftc -O bench/amx/amx_bnns_graph.swift -o /tmp/amxbench/bnns_graph
// Run (one shape or all):
//   bnns_graph <model-dir> [tag]      # tag omitted -> all 12 shapes
import Accelerate
import Foundation

let M = 128                       // S, prefill batch (paper headline)
let TRIALS = 30
let WARMUP = 3

// (tag, N, K): C[M,N] = X[M,K] @ W[K,N]
let SHAPES: [(String, Int, Int)] = [
    ("gpt2_qkv",2048,2048),("gpt2_ffn1",8192,2048),("gpt2_ffn2",2048,8192),("gpt2_lmh",60000,2048),
    ("tiny_qkv",2048,2048),("tiny_ffn1",5632,2048),("tiny_ffn2",2048,5632),("tiny_lmh",32000,2048),
    ("llama_qkv",4096,4096),("llama_ffn1",11008,4096),("llama_ffn2",4096,11008),("llama_lmh",32000,4096),
]

func now() -> Double { Double(DispatchTime.now().uptimeNanoseconds) * 1e-9 }

@available(macOS 15.0, *)
func runShape(dir: String, tag: String, N: Int, K: Int) {
    let pkg = "\(dir)/\(tag).mlmodelc"   // BNNS Graph compiles the .mlmodelc, not the raw .mlpackage
    let wPath = "\(dir)/\(tag)_W.f32"

    // --- compile + repack weight ONCE (amortized, untimed) ---
    let opts = BNNSGraph.CompileOptions()
    let ctx: BNNSGraph.Context
    do {
        ctx = try BNNSGraph.Context(compileFromPath: pkg, functionName: nil, options: opts)
    } catch {
        print("\(tag): compile FAILED: \(error)"); return
    }
    let fn: String? = ctx.functionNames.first
    let names = ctx.argumentNames(forFunction: fn)

    // host buffers: input X [M,K], output Y [M,N]
    let X = UnsafeMutableBufferPointer<Float>.allocate(capacity: M*K)
    let Y = UnsafeMutableBufferPointer<Float>.allocate(capacity: M*N)
    defer { X.deallocate(); Y.deallocate() }
    var seed: UInt64 = 0x9E3779B97F4A7C15 &+ UInt64(tag.count)
    func rnd() -> Float { seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17
        return Float(Int32(truncatingIfNeeded: seed)) * (1.0/Float(Int32.max)) }
    for i in 0..<M*K { X[i] = rnd() }
    for i in 0..<M*N { Y[i] = 0 }

    // map each graph argument to our buffer by element count (input=M*K, output=M*N)
    var args = [BNNSTensor](repeating: BNNSTensor(), count: names.count)
    for name in names {
        let pos = ctx.argumentPosition(forFunction: fn, argument: name)
        guard let probe = ctx.allocateTensor(forFunction: fn, argument: name,
                                             fillKnownDynamicShapes: true) else {
            print("\(tag): allocateTensor failed for \(name)"); return
        }
        if probe.count == M*K {
            args[pos] = BNNSTensor(data: UnsafeMutableRawBufferPointer(X),
                                   shape: [M,K], stride: [K,1], dataType: .float)
        } else {
            args[pos] = BNNSTensor(data: UnsafeMutableRawBufferPointer(Y),
                                   shape: [M,N], stride: [N,1], dataType: .float)
        }
    }

    // --- timed: executeFunction only ---
    for _ in 0..<WARMUP { try? ctx.executeFunction(fn, arguments: &args) }
    var best = Double.greatestFiniteMagnitude
    var times = [Double]()
    for _ in 0..<TRIALS {
        let t0 = now()
        do { try ctx.executeFunction(fn, arguments: &args) }
        catch { print("\(tag): execute FAILED: \(error)"); return }
        let dt = now() - t0
        times.append(dt); best = min(best, dt)
    }
    times.sort()
    let med = times[times.count/2]
    let flop = 2.0 * Double(M) * Double(N) * Double(K)
    let gMed = flop / med * 1e-9
    let gBest = flop / best * 1e-9

    // --- bit-accuracy check vs cblas_sgemm on the same baked W ---
    var maxAbs: Float = .nan
    if let wd = FileManager.default.contents(atPath: wPath) {
        let Wf = UnsafeMutableBufferPointer<Float>.allocate(capacity: K*N); defer { Wf.deallocate() }
        _ = wd.copyBytes(to: UnsafeMutableRawBufferPointer(Wf), count: K*N*4)
        let Cref = UnsafeMutableBufferPointer<Float>.allocate(capacity: M*N); defer { Cref.deallocate() }
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, Int32(M), Int32(N), Int32(K),
                    1.0, X.baseAddress, Int32(K), Wf.baseAddress, Int32(N), 0.0, Cref.baseAddress, Int32(N))
        maxAbs = 0
        for i in 0..<M*N { maxAbs = max(maxAbs, abs(Y[i] - Cref[i])) }
    }
    print(String(format: "%-12@ N=%5d K=%5d  med %6.0f  best %6.0f GFLOPS  max-abs-diff %.3e",
                 tag as NSString, N, K, gMed, gBest, maxAbs))
    print(String(format: "CSV,%@,%.1f,%.1f,%.3e", tag as NSString, gMed, gBest, maxAbs))
}

// ---- main ----
guard #available(macOS 15.0, *) else { print("needs macOS 15+"); exit(1) }
let argv = CommandLine.arguments
guard argv.count >= 2 else { print("usage: bnns_graph <model-dir> [tag]"); exit(2) }
let dir = argv[1]
let only = argv.count >= 3 ? argv[2] : nil
print("# BNNS Graph fp32 (compile+repack once, execute timed)  M=\(M)")
for (tag,N,K) in SHAPES where only == nil || only == tag {
    runShape(dir: dir, tag: tag, N: N, K: K)
}

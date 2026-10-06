// Copyright © 2026 Apple Inc.

import MLX
import XCTest

/// Metal threadgroup safety regressions for the public MLX API.
///
/// These tests launch real Metal kernels. They require a physical GPU. A test
/// that does not execute, or a GPU that is absent, is not a pass.
///
/// The compiled pipeline ceiling cannot be forced from Swift. Agreement with
/// the CPU implementation checks the numeric result of whichever kernel the
/// GPU selects. It does not prove that a constrained pipeline took the looped
/// RMSNorm kernel, the 2-pass vector attention kernel, or a block below 1024.
final class MetalThreadgroupSafetyTests: XCTestCase {

    /// Historical default-device value from `MLXRandomTests.testUniformSingle`.
    func testRandomBitsKnownValueAndEmpty() {
        let key = MLXRandom.key(0)
        let scalar = MLXRandom.uniform(0 ..< 10, key: key, stream: .gpu).item(Float.self)
        XCTAssertEqual(scalar, 4.18, accuracy: 0.01)

        let empty = MLXRandom.uniform(0.0 ..< 1.0, [0, 4], key: key, stream: .gpu)
        MLX.eval(empty)
        XCTAssertEqual(empty.shape, [0, 4])
        XCTAssertEqual(empty.size, 0)

        let wideKey = MLXRandom.key(1)
        let first = MLXRandom.uniform(0.0 ..< 1.0, [64, 128], key: wideKey, stream: .gpu)
        let second = MLXRandom.uniform(0.0 ..< 1.0, [64, 128], key: wideKey, stream: .gpu)
        XCTAssertEqual(first.shape, [64, 128])
        assertEqual(first, second)
    }

    func testGatherMatchesCPUAndKeepsEmptyShape() {
        let table = MLXArray(0 ..< (128 * 64), [128, 64])
        let rows = MLXArray(stride(from: 0, through: 127, by: 3))
        assertEqual(
            table.take(rows, axis: 0, stream: .gpu),
            table.take(rows, axis: 0, stream: .cpu))

        let columns = MLXArray([1, 2, 7, 30])
        let viewed = table.transposed(stream: .gpu)
        assertEqual(
            viewed.take(columns, axis: 1, stream: .gpu),
            viewed.take(columns, axis: 1, stream: .cpu))

        let cube = MLXArray(0 ..< (8 * 16 * 32), [8, 16, 32])
        // Outer index shapes [3, 1] and [1, 4] broadcast. Each take keeps its
        // own axis. There is no public gather that both accepts these indices
        // and forwards .cpu/.gpu; advanced subscript does not forward stream.
        let i0 = MLXArray([0, 3, 7], [3, 1])
        let i1 = MLXArray([1, 4, 9, 15], [1, 4])
        assertEqual(
            cube.take(i0, axis: 0, stream: .gpu),
            cube.take(i0, axis: 0, stream: .cpu))
        assertEqual(
            cube.take(i1, axis: 1, stream: .gpu),
            cube.take(i1, axis: 1, stream: .cpu))

        // Rank-zero index on a non-contiguous axis-1 take. The front gather
        // path does not apply, so the general gather binds empty index metadata.
        let scalar = MLXArray(Int32(2))
        let grid = MLXArray(0 ..< 24, [4, 6]).transposed()
        assertEqual(
            grid.take(scalar, axis: 1, stream: .gpu),
            grid.take(scalar, axis: 1, stream: .cpu))

        let emptyRows = zeros([0], dtype: .int32)
        let taken = cube.take(emptyRows, axis: 0, stream: .gpu)
        MLX.eval(taken)
        XCTAssertEqual(taken.shape, [0, 16, 32])
        XCTAssertEqual(taken.size, 0)
    }

    func testRMSNormMatchesCPUAcrossKernelSplit() {
        let eps: Float = 1e-5
        // 64 stays on the single-row kernel and, with 600 rows, uses the
        // grouped VJP. 4096 asks that kernel for 1024 threads. 4224 is past
        // RMS_LOOPED_LIMIT and uses the looped kernel.
        for shape in [[600, 64], [2, 4096], [2, 4224]] {
            let count = shape.reduce(1, *)
            let values = (0 ..< count).map { Float($0 % 19) * 0.05 - 0.4 }
            let x = MLXArray(values, shape)
            let weight = ones([shape[1]])
            assertEqual(
                rmsNorm(x, weight: weight, eps: eps, stream: .gpu),
                rmsNorm(x, weight: weight, eps: eps, stream: .cpu),
                rtol: 1e-3,
                atol: 1e-3)

            let gpuGrad = grad { (x: MLXArray) -> MLXArray in
                rmsNorm(x, weight: weight, eps: eps, stream: .gpu).sum(stream: .gpu)
            }(x)
            let cpuGrad = grad { (x: MLXArray) -> MLXArray in
                rmsNorm(x, weight: weight, eps: eps, stream: .cpu).sum(stream: .cpu)
            }(x)
            assertEqual(gpuGrad, cpuGrad, rtol: 1e-3, atol: 1e-3)
        }
    }

    /// Input and weight gradients for wide rows. Value equality does not show
    /// which RMSNorm kernel the GPU selected.
    func testRMSNormInputAndWeightGradients() {
        let eps: Float = 1e-5
        for shape in [[600, 4096], [600, 4224]] {
            let count = shape.reduce(1, *)
            let values = (0 ..< count).map { Float($0 % 19) * 0.05 - 0.4 }
            let x = MLXArray(values, shape)
            let axis = shape[1]
            let weight = MLXArray((0 ..< axis).map { Float($0 % 7) * 0.1 + 0.5 })
            let differentiate = { (stream: StreamOrDevice) in
                grad(
                    { (inputs: [MLXArray]) -> [MLXArray] in
                        [
                            rmsNorm(
                                inputs[0], weight: inputs[1], eps: eps, stream: stream
                            ).sum(stream: stream)
                        ]
                    }, argumentNumbers: [0, 1])([x, weight])
            }
            let gpuGrad = differentiate(.gpu)
            let cpuGrad = differentiate(.cpu)
            XCTAssertEqual(gpuGrad.count, 2)
            assertEqual(gpuGrad[0], cpuGrad[0], rtol: 1e-3, atol: 1e-3)
            assertEqual(gpuGrad[1], cpuGrad[1], rtol: 1e-3, atol: 1e-3)
            XCTAssertEqual(gpuGrad[0].shape, shape)
            XCTAssertEqual(gpuGrad[1].shape, [axis])
        }
    }

    func testRoPEMatchesCPU() {
        let single = sequence(count: 2 * 4 * 1 * 32, shape: [2, 4, 1, 32])
        assertEqual(
            MLXFast.RoPE(
                single, dimensions: 32, traditional: false, base: 10_000, scale: 1, offset: 3,
                stream: .gpu),
            MLXFast.RoPE(
                single, dimensions: 32, traditional: false, base: 10_000, scale: 1, offset: 3,
                stream: .cpu),
            rtol: 1e-4,
            atol: 1e-4)

        let batched = sequence(count: 2 * 4 * 8 * 64, shape: [2, 4, 8, 64])
        assertEqual(
            MLXFast.RoPE(
                batched, dimensions: 32, traditional: true, base: 10_000, scale: 1, offset: 1,
                stream: .gpu),
            MLXFast.RoPE(
                batched, dimensions: 32, traditional: true, base: 10_000, scale: 1, offset: 1,
                stream: .cpu),
            rtol: 1e-4,
            atol: 1e-4)
    }

    func testVectorAttentionMatchesCPU() {
        let queries = sequence(count: 64, shape: [1, 1, 1, 64])
        let keys = sequence(count: 16 * 64, shape: [1, 1, 16, 64])
        let values = sequence(count: 16 * 64, shape: [1, 1, 16, 64], phase: 3)
        assertEqual(
            MLXFast.scaledDotProductAttention(
                queries: queries, keys: keys, values: values, scale: 0.125, mask: .none,
                stream: .gpu),
            MLXFast.scaledDotProductAttention(
                queries: queries, keys: keys, values: values, scale: 0.125, mask: .none,
                stream: .cpu),
            rtol: 1e-3,
            atol: 1e-3)
    }

    func testGeneralDispatchMatchesCPU() {
        let source = MLXArray(0 ..< 81, [3, 3, 3, 3])
        let lhs = source.transposed(1, 0, 3, 2)
        let rhs = source.transposed(2, 3, 0, 1)
        assertEqual(
            MLX.add(lhs, rhs, stream: .gpu),
            MLX.add(lhs, rhs, stream: .cpu))
        assertEqual(
            negative(lhs, stream: .gpu),
            negative(lhs, stream: .cpu))
        let condition = MLX.greater(lhs, rhs, stream: .gpu)
        assertEqual(
            `where`(condition, lhs, rhs, stream: .gpu),
            `where`(condition, lhs, rhs, stream: .cpu))
        assertEqual(
            lhs.asType(.float32, stream: .gpu),
            lhs.asType(.float32, stream: .cpu))
    }
}

private func sequence(count: Int, shape: [Int], phase: Int = 1) -> MLXArray {
    let values = (0 ..< count).map { Float(($0 * phase) % 23) * 0.03 - 0.2 }
    return MLXArray(values, shape)
}

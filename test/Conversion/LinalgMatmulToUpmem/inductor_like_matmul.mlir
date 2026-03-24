// RUN: triton-shared-opt --linalg-matmul-to-upmem %s | FileCheck %s

module {
  func.func @inductor_like(%arg0: memref<*xf32> loc("arg_A"), %arg1: memref<*xf32> loc("arg_B"), %arg2: memref<*xf32> loc("out_ptr0"), %arg3: i32 loc("m"), %arg4: i32 loc("n"), %arg5: i32 loc("k")) {
    %a_cast = memref.reinterpret_cast %arg0 to offset: [0], sizes: [64, 16], strides: [16, 1] : memref<*xf32> to memref<64x16xf32, strided<[16, 1], offset: ?>>
    %a_tensor = bufferization.to_tensor %a_cast restrict : memref<64x16xf32, strided<[16, 1], offset: ?>> to tensor<64x16xf32>
    %b_cast = memref.reinterpret_cast %arg1 to offset: [0], sizes: [16, 32], strides: [32, 1] : memref<*xf32> to memref<16x32xf32, strided<[32, 1], offset: ?>>
    %alloc = memref.alloc() : memref<16x32xf32>
    memref.copy %b_cast, %alloc : memref<16x32xf32, strided<[32, 1], offset: ?>> to memref<16x32xf32>
    %b_tensor = bufferization.to_tensor %alloc restrict writable : memref<16x32xf32> to tensor<16x32xf32>
    %c0 = arith.constant 0.0 : f32
    %out_init = tensor.empty() : tensor<64x32xf32>
    %filled = linalg.fill ins(%c0 : f32) outs(%out_init : tensor<64x32xf32>) -> tensor<64x32xf32>
    %matmul = linalg.matmul ins(%a_tensor, %b_tensor : tensor<64x16xf32>, tensor<16x32xf32>) outs(%filled : tensor<64x32xf32>) -> tensor<64x32xf32>
    %reinterpret = memref.reinterpret_cast %arg2 to offset: [0], sizes: [64, 32], strides: [32, 1] : memref<*xf32> to memref<64x32xf32, strided<[32, 1], offset: ?>>
    bufferization.materialize_in_destination %matmul in writable %reinterpret : (tensor<64x32xf32>, memref<64x32xf32, strided<[32, 1], offset: ?>>) -> ()
    return
  }
}

// CHECK: func.func @inductor_like
// CHECK-SAME: attributes {
// CHECK-DAG: upmem.a_ptr_idx = 0 : i32
// CHECK-DAG: upmem.b_ptr_idx = 1 : i32
// CHECK-DAG: upmem.c_ptr_idx = 2 : i32
// CHECK-DAG: upmem.m_idx = 3 : i32
// CHECK-DAG: upmem.n_idx = 4 : i32
// CHECK-DAG: upmem.k_idx = 5 : i32
// CHECK-DAG: upmem.bm = 64 : i32
// CHECK-DAG: upmem.bk = 16 : i32
// CHECK-DAG: upmem.bn = 32 : i32
// CHECK-DAG: upmem.elem_type = "f32"
// CHECK-DAG: upmem.acc_type = "f32"

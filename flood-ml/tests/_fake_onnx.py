# tiny fake onnx models with the real i/o names, for tests
from __future__ import annotations

import math

import numpy as np
import onnx
from onnx import TensorProto, helper

SIZE = 224
GRID = 7
POOL = SIZE // GRID


def logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def build_stage_onnx(
    path,
    *,
    prob_bias: float = 0.0,
    prob_slope: float = 0.0,
    cam_mode: str = "constant",
    cam_bias: float = 0.0,
    cam_scale: float = 1.0,
    opset: int = 17,
):
    image = helper.make_tensor_value_info("image", TensorProto.FLOAT, ["N", SIZE, SIZE, 3])
    prob_out = helper.make_tensor_value_info("prob", TensorProto.FLOAT, ["N", 1])
    cam_out = helper.make_tensor_value_info("cam", TensorProto.FLOAT, ["N", GRID, GRID])

    nodes = []
    inits = []

    nodes.append(helper.make_node("ReduceMean", ["image"], ["mean_all"], axes=[1, 2, 3], keepdims=0))
    inits.append(helper.make_tensor("c255", TensorProto.FLOAT, [], [255.0]))
    nodes.append(helper.make_node("Div", ["mean_all", "c255"], ["mean_norm"]))
    inits.append(helper.make_tensor("prob_slope", TensorProto.FLOAT, [], [prob_slope]))
    inits.append(helper.make_tensor("prob_bias", TensorProto.FLOAT, [], [prob_bias]))
    nodes.append(helper.make_node("Mul", ["mean_norm", "prob_slope"], ["scaled"]))
    nodes.append(helper.make_node("Add", ["scaled", "prob_bias"], ["logit"]))
    nodes.append(helper.make_node("Sigmoid", ["logit"], ["prob_flat"]))
    inits.append(helper.make_tensor("prob_shape", TensorProto.INT64, [2], [-1, 1]))
    nodes.append(helper.make_node("Reshape", ["prob_flat", "prob_shape"], ["prob"]))

    if cam_mode == "constant":
        nodes.append(helper.make_node("Shape", ["mean_all"], ["n_dim"]))
        inits.append(helper.make_tensor("grid_shape", TensorProto.INT64, [2], [GRID, GRID]))
        nodes.append(helper.make_node("Concat", ["n_dim", "grid_shape"], ["cam_target_shape"], axis=0))
        inits.append(helper.make_tensor(
            "cam_const", TensorProto.FLOAT, [GRID, GRID], [cam_bias] * (GRID * GRID),
        ))
        nodes.append(helper.make_node("Expand", ["cam_const", "cam_target_shape"], ["cam_pre"]))
    else:
        nodes.append(helper.make_node("ReduceMean", ["image"], ["gray"], axes=[3], keepdims=0))
        inits.append(helper.make_tensor(
            "pool_shape", TensorProto.INT64, [5], [-1, GRID, POOL, GRID, POOL],
        ))
        nodes.append(helper.make_node("Reshape", ["gray", "pool_shape"], ["gray_grid"]))
        nodes.append(helper.make_node("ReduceMean", ["gray_grid"], ["pooled"], axes=[2, 4], keepdims=0))
        inits.append(helper.make_tensor("c255b", TensorProto.FLOAT, [], [255.0]))
        nodes.append(helper.make_node("Div", ["pooled", "c255b"], ["pooled_norm"]))
        inits.append(helper.make_tensor("cam_scale", TensorProto.FLOAT, [], [cam_scale]))
        inits.append(helper.make_tensor("cam_bias", TensorProto.FLOAT, [], [cam_bias]))
        nodes.append(helper.make_node("Mul", ["pooled_norm", "cam_scale"], ["cam_scaled"]))
        nodes.append(helper.make_node("Add", ["cam_scaled", "cam_bias"], ["cam_pre"]))

    nodes.append(helper.make_node("Relu", ["cam_pre"], ["cam"]))

    graph = helper.make_graph(nodes, "fake_stage", [image], [cam_out, prob_out], initializer=inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    model.ir_version = 9
    onnx.checker.check_model(model)
    onnx.save(model, str(path))


def uniform_image(value: int, size=(SIZE + 32, SIZE + 32)) -> np.ndarray:
    arr = np.full((size[1], size[0], 3), value, dtype=np.uint8)
    return arr


def gradient_image(size=(SIZE + 32, SIZE + 32)) -> np.ndarray:
    w, h = size
    xs = np.linspace(0, 255, w, dtype=np.float32)
    grid = np.tile(xs, (h, 1))
    arr = np.stack([grid, grid, grid], axis=-1).astype(np.uint8)
    return arr


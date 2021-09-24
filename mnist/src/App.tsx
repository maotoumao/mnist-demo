import React, { useEffect, useRef, useState } from "react";
import * as ort from "onnxruntime-web";
import pica from "pica";
import model from "./model/mnist.onnx?url";
import "./App.css";

interface Point {
  x: number;
  y: number;
}

function drawLine(
  context: CanvasRenderingContext2D,
  prevPoint: Point,
  endPoint: Point
) {
  context.moveTo(prevPoint.x, prevPoint.y);
  context.lineTo(endPoint.x, endPoint.y);
  context.stroke();
}

function App() {
  const [label, setLabel] = useState<string>();
  const isPainting = useRef<boolean>(false);
  const prevPoint = useRef<Point>();
  const canvasRef = useRef<any>();
  const onnxSession = useRef<any>();

  // bind canvas events
  useEffect(() => {
    if (!canvasRef.current) {
      return;
    }
    const ctx: CanvasRenderingContext2D = canvasRef.current.getContext("2d");
    ctx.lineWidth = 30;
    ctx.strokeStyle = "#000";
    canvasRef.current.onmousedown = function (e: MouseEvent) {
      ctx.beginPath();
      let [x, y] = [e.offsetX, e.offsetY];
      isPainting.current = true;
      prevPoint.current = {
        x,
        y,
      };
    };

    canvasRef.current.onmousemove = function (e: MouseEvent) {
      let [x, y] = [e.offsetX, e.offsetY];
      if (isPainting.current && prevPoint.current) {
        drawLine(ctx, prevPoint.current, { x, y });
        prevPoint.current = {
          x,
          y,
        };
      }
    };

    canvasRef.current.onmouseup = function () {
      isPainting.current = false;
      ctx.closePath();
    };

    canvasRef.current.onmouseout = function () {
      isPainting.current = false;
      ctx.closePath();
    };

    canvasRef.current.ontouchstart = function (e: TouchEvent) {
      ctx.beginPath();
      let [x, y] = [e.touches[0].pageX - canvasRef.current.offsetLeft, e.touches[0].clientY - canvasRef.current.offsetTop];
      isPainting.current = true;
      prevPoint.current = {
        x,
        y,
      };
    };

    canvasRef.current.ontouchmove = function (e: TouchEvent) {
      let [x, y] = [e.touches[0].pageX - canvasRef.current.offsetLeft, e.touches[0].clientY - canvasRef.current.offsetTop];
      if (isPainting.current && prevPoint.current) {
        drawLine(ctx, prevPoint.current, { x, y });
        prevPoint.current = {
          x,
          y,
        };
      }
    };

    canvasRef.current.ontouchend = function () {
      isPainting.current = false;
      ctx.closePath();
    };

    canvasRef.current.ontouchout = function () {
      isPainting.current = false;
      ctx.closePath();
    };
  }, []);

  useEffect(() => {
    document.body.addEventListener(
      "touchmove",
      (e) => {
        e.preventDefault();
      },
      {
        passive: false,
      }
    );
    // load model
    (async () => {
      const session = await ort.InferenceSession.create(model);
      onnxSession.current = session;
    })();
  }, []);

  async function onPredict() {
    if (!canvasRef.current) {
      return;
    }
    if (!onnxSession.current) {
      alert("模型还没加载完");
      return;
    }
    const ctx: CanvasRenderingContext2D = canvasRef.current.getContext("2d");
    const data = ctx.getImageData(0, 0, 300, 300);
    const resizedBuffer = await pica().resizeBuffer({
      // @ts-ignore
      src: data.data,
      alpha: true,
      width: 300,
      height: 300,
      toWidth: 28,
      toHeight: 28,
    });
    // 每四个一组 转化为0-255
    let _result = [];
    for (let i = 0; i < resizedBuffer.length; i += 4) {
      _result.push(resizedBuffer[i + 3] < 64 ? -1 : 1);
    }
    const inputArray = Float32Array.from(_result);

    const inputs = new ort.Tensor("float32", inputArray, [1, 1, 28, 28]);

    try {
      const outputs = await onnxSession.current.run({
        input: inputs,
      });
      console.log(outputs.output.data);
      let d = outputs.output.data.findIndex(
        (item: any) => item === Math.max(...outputs.output.data)
      );
      setLabel(
        `${d}  概率: ${(Math.exp(outputs.output.data[d]) * 100).toFixed(2)}%`
      );
    } catch (e) {
      console.log(e);
    }
  }

  function onClear() {
    if (!canvasRef.current) {
      return;
    }
    const ctx: CanvasRenderingContext2D = canvasRef.current.getContext("2d");
    ctx.clearRect(0, 0, 300, 300);
  }

  return (
    <div className="app">
      <h1>Mnist Demo</h1>
      <canvas
        ref={(ref) => (canvasRef.current = ref)}
        id="canvas"
        width={300}
        height={300}
      ></canvas>
      <div className="btn-group">
        <button onClick={onPredict}>识别</button>
        <button onClick={onClear}>清空</button>
      </div>
      {typeof label === "string" && <div>识别结果: {label ?? ""}</div>}
    </div>
  );
}

export default App;

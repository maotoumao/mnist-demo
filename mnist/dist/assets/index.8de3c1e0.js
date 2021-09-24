import{j as e,r as t,p as r,R as n,a as o}from"./vendor.136abdb5.js";!function(){const e=document.createElement("link").relList;if(!(e&&e.supports&&e.supports("modulepreload"))){for(const e of document.querySelectorAll('link[rel="modulepreload"]'))t(e);new MutationObserver((e=>{for(const r of e)if("childList"===r.type)for(const e of r.addedNodes)"LINK"===e.tagName&&"modulepreload"===e.rel&&t(e)})).observe(document,{childList:!0,subtree:!0})}function t(e){if(e.ep)return;e.ep=!0;const t=function(e){const t={};return e.integrity&&(t.integrity=e.integrity),e.referrerpolicy&&(t.referrerPolicy=e.referrerpolicy),"use-credentials"===e.crossorigin?t.credentials="include":"anonymous"===e.crossorigin?t.credentials="omit":t.credentials="same-origin",t}(e);fetch(e.href,t)}}();const c=e.exports.jsx,u=e.exports.jsxs,s=window.ort;function i(e,t,r){e.moveTo(t.x,t.y),e.lineTo(r.x,r.y),e.stroke()}function a(){const[e,n]=t.exports.useState(),o=t.exports.useRef(!1),a=t.exports.useRef(),f=t.exports.useRef(),l=t.exports.useRef();return t.exports.useEffect((()=>{if(!f.current)return;const e=f.current.getContext("2d");e.lineWidth=30,e.strokeStyle="#000",f.current.onmousedown=function(t){e.beginPath();let[r,n]=[t.offsetX,t.offsetY];o.current=!0,a.current={x:r,y:n}},f.current.onmousemove=function(t){let[r,n]=[t.offsetX,t.offsetY];o.current&&a.current&&(i(e,a.current,{x:r,y:n}),a.current={x:r,y:n})},f.current.onmouseup=function(){o.current=!1,e.closePath()},f.current.onmouseout=function(){o.current=!1,e.closePath()},f.current.ontouchstart=function(t){e.beginPath();let[r,n]=[t.touches[0].pageX-f.current.offsetLeft,t.touches[0].clientY-f.current.offsetTop];o.current=!0,a.current={x:r,y:n}},f.current.ontouchmove=function(t){let[r,n]=[t.touches[0].pageX-f.current.offsetLeft,t.touches[0].clientY-f.current.offsetTop];o.current&&a.current&&(i(e,a.current,{x:r,y:n}),a.current={x:r,y:n})},f.current.ontouchend=function(){o.current=!1,e.closePath()},f.current.ontouchout=function(){o.current=!1,e.closePath()}}),[]),t.exports.useEffect((()=>{document.body.addEventListener("touchmove",(e=>{e.preventDefault()}),{passive:!1}),(async()=>{const e=await s.InferenceSession.create("/assets/mnist.c6809106.onnx");l.current=e})()}),[]),u("div",{className:"app",children:[c("h1",{children:"Mnist Demo"}),c("canvas",{ref:e=>f.current=e,id:"canvas",width:300,height:300}),u("div",{className:"btn-group",children:[c("button",{onClick:async function(){if(!f.current)return;if(!l.current)return void alert("模型还没加载完");const e=f.current.getContext("2d").getImageData(0,0,300,300),t=await r().resizeBuffer({src:e.data,alpha:!0,width:300,height:300,toWidth:28,toHeight:28});let o=[];for(let r=0;r<t.length;r+=4)o.push(t[r+3]<64?-1:1);const c=Float32Array.from(o),u=new s.Tensor("float32",c,[1,1,28,28]);try{const e=await l.current.run({input:u});console.log(e.output.data);let t=e.output.data.findIndex((t=>t===Math.max(...e.output.data)));n(`${t}  概率: ${(100*Math.exp(e.output.data[t])).toFixed(2)}%`)}catch(i){console.log(i)}},children:"识别"}),c("button",{onClick:function(){if(!f.current)return;f.current.getContext("2d").clearRect(0,0,300,300)},children:"清空"})]}),"string"==typeof e&&u("div",{children:["识别结果: ",null!=e?e:""]})]})}n.render(c(o.StrictMode,{children:c(a,{})}),document.getElementById("root"));

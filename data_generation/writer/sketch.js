const WIDTH = 400;
const HEIGHT = 400;
const NUM_CARS = 10;


function setup() {  
  createCanvas(WIDTH, HEIGHT);
  // background(getRandomColor());
  angleMode("radians");
}

function draw() {
  rect(30, 20, 55, 100, 20);
}

const colors = [
  [175, 172, 153],
  [186, 184, 173],
  [178, 176, 166],
  [215, 218, 214],
  [181, 174, 161],
  [212, 212, 205],
  [191, 191, 186],
  [188, 186, 177],
  [74, 75, 69],
  [184, 181, 166],
  [176, 173, 161],
  [186, 188, 183],
  [91, 89, 63],
  [0, 0, 0],
  [207, 203, 189],
  [192, 192, 184],
  [197, 194, 183],
  [191, 188, 175],
  [169, 164, 147],
  [207, 206, 202],
  [177, 175, 162],
  [182, 177, 165],
  [54, 67, 66],
  [186, 182, 169],
  [185, 181, 168],
  [173, 170, 157],
  [72, 78, 74],
  [213, 207, 193],
  [189, 186, 174],
];

function getSize() {
  const color = colors[Math.floor(Math.random() * colors.length)];
  return color;
}

function getHoodSize() {
  const color = colors[Math.floor(Math.random() * colors.length)];
  return color;
}

function getRooftopWindowSize() {
  const color = colors[Math.floor(Math.random() * colors.length)];
  return color;
}

function getRandomLines() {
  const color = colors[Math.floor(Math.random() * colors.length)];
  return color;
}

function getColor() {
  const color = colors[Math.floor(Math.random() * colors.length)];
  return color;
}
